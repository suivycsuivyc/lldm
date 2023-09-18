import inspect
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel, UNet2DConditionModel
from diffusers import DDPMScheduler, DDIMScheduler, PNDMScheduler
from diffusers import AutoencoderKL
from diffusers.utils.torch_utils import randn_tensor

import models
from models.unet import UNet2DConditionModelWrapped
from models import register


HF_CLS_DICT_UNET = {
    'unet2d': UNet2DModel,
    'unet2d_cond': UNet2DConditionModel,
    'unet2d_cond_wrapped': UNet2DConditionModelWrapped,
}
HF_CLS_DICT_SCHEDULER = {
    'ddpm': DDPMScheduler,
    'ddim': DDIMScheduler,
    'pndm': PNDMScheduler,
}
HF_CLS_DICT_VAE = {
    'ae_kl': AutoencoderKL,
}
HF_CLS_DICT = dict()
HF_CLS_DICT.update(HF_CLS_DICT_UNET)
HF_CLS_DICT.update(HF_CLS_DICT_SCHEDULER)
HF_CLS_DICT.update(HF_CLS_DICT_VAE)


def make_hf_object(spec):
    if spec is None:
        return None
    name = spec['name']
    from_pretrained = spec['args'].pop('from_pretrained', None)
    CLS = HF_CLS_DICT[name]
    if from_pretrained is not None:
        if name in set(HF_CLS_DICT_UNET.keys()):
            kwargs = {'subfolder': 'unet', 'use_safetensors': True}
        elif name in set(HF_CLS_DICT_SCHEDULER.keys()):
            kwargs = {'subfolder': 'scheduler'}
        elif name in set(HF_CLS_DICT_VAE.keys()):
            kwargs = {'subfolder': 'vae', 'use_safetensors': True}
        else:
            raise NotImplementedError
        obj = CLS.from_pretrained(from_pretrained, **kwargs)
    else:
        obj = CLS(**spec['args'])
    return obj


@register('diffusion_model')
class DiffusionModel(nn.Module):

    def __init__(self, unet, scheduler, vae=None, unet_ema_rate=0.9999, train_scheduler=None):
        super().__init__()

        self.unet = make_hf_object(unet) if unet['name'] in set(HF_CLS_DICT_UNET.keys()) else models.make(unet)
        self.unet_ema_rate = unet_ema_rate
        if unet_ema_rate is not None:
            self.unet_ema = copy.deepcopy(self.unet)
            self.unet_ema.requires_grad_(False)

        self.scheduler = make_hf_object(scheduler)
        if train_scheduler is None:
            train_scheduler = scheduler
        self.train_scheduler = make_hf_object(train_scheduler)

        self.vae = make_hf_object(vae)
        if self.vae is not None:
            self.vae.requires_grad_(False)
            self.vae.eval()
            if isinstance(self.vae, AutoencoderKL):
                self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            else:
                raise NotImplementedError

    def compute_snr(self, timesteps, scheduler):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def get_losses(self, x, cond=None, input_perturbation=0, prediction_type=None, snr_gamma=None):
        if self.vae is not None:
            x = self.vae.encode(x)['latent_dist'].sample()
            x = x * self.vae.config.scaling_factor

        scheduler = self.train_scheduler

        noise = torch.randn_like(x)
        if input_perturbation:
            new_noise = noise + input_perturbation * torch.randn_like(noise)
        bsz = x.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=x.device)
        timesteps = timesteps.long()

        # Add noise to the x according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        if input_perturbation:
            noisy_x = scheduler.add_noise(x, new_noise, timesteps)
        else:
            noisy_x = scheduler.add_noise(x, noise, timesteps)

        # Get the target for loss depending on the prediction type
        if prediction_type is not None:
            # set prediction_type of scheduler if defined
            scheduler.register_to_config(prediction_type=prediction_type)

        if scheduler.config.prediction_type == 'epsilon':
            target = noise
        elif scheduler.config.prediction_type == 'v_prediction':
            target = scheduler.get_velocity(x, noise, timesteps)
        else:
            raise ValueError(f'Unknown prediction type {scheduler.config.prediction_type}')

        # Predict the noise residual and compute loss
        if cond is None:
            cond = {}
        model_pred = self.unet(noisy_x, timesteps, **cond)['sample']

        if snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction='mean')
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = self.compute_snr(timesteps, scheduler)
            mse_loss_weights = (
                torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            # We first calculate the original loss. Then we mean over the non-batch dimensions and
            # rebalance the sample-wise losses with their respective loss weights.
            # Finally, we take the mean of the rebalanced loss.
            loss = F.mse_loss(model_pred.float(), target.float(), reduction='none')
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        return loss

    @torch.no_grad()
    def sample(self, batch_size=1, shape=None, num_steps=50, cond=None, cond_neg=None, guidance_scale=1.0, use_unet_ema=False,
               num_images_per_cond=1, eta=0.0, guidance_rescale=0.0, sample=None, generator=None, output_type='image'):
        device = next(self.unet.parameters()).device
        dtype = next(self.unet.parameters()).dtype

        # Here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            cond = {k: torch.cat([cond[k], cond_neg[k]]) for k in cond.keys()}

        # Prepare timesteps
        self.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare sample variables
        if shape is None:
            if isinstance(self.unet, UNet2DModel) or isinstance(self.unet, UNet2DConditionModel):
                shape = (self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size)
            else:
                shape = self.unet.config.sample_shape
        shape = (batch_size * num_images_per_cond,) + tuple(shape)
        assert not (isinstance(generator, list) and len(generator) != batch_size * num_images_per_cond)
        if sample is None:
            sample = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            sample = sample.to(device)

        # Scale the initial noise by the standard deviation required by the scheduler
        sample = sample * self.scheduler.init_noise_sigma

        # Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self._prepare_extra_step_kwargs(generator, eta)

        # Denoising loop
        for i, t in enumerate(timesteps):
            # Expand the sample if we are doing classifier free guidance
            unet_input = torch.cat([sample] * 2) if do_classifier_free_guidance else sample
            unet_input = self.scheduler.scale_model_input(unet_input, t)

            # Predict the noise residual
            if cond is None:
                cond = {}
            unet = self.unet_ema if use_unet_ema else self.unet
            noise_pred = unet(unet_input, t, **cond)['sample']

            # Perform guidance
            if do_classifier_free_guidance:
                noise_pred_cond, noise_pred_cond_neg = noise_pred.chunk(2)
                noise_pred = noise_pred_cond_neg + guidance_scale * (noise_pred_cond - noise_pred_cond_neg)

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = self._rescale_noise_cfg(noise_pred, noise_pred_cond, guidance_rescale=guidance_rescale)

            # Compute the previous noisy sample x_t -> x_t-1
            sample = self.scheduler.step(noise_pred, t, sample, **extra_step_kwargs, return_dict=False)[0]

        if self.vae is not None and output_type == 'image':
            ret = self.vae.decode(sample / self.vae.config.scaling_factor)['sample']
        elif output_type == 'latent':
            ret = sample
        else:
            raise NotImplementedError
        return ret

    def _prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = 'eta' in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs['eta'] = eta
        # check if the scheduler accepts generator
        accepts_generator = 'generator' in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs['generator'] = generator
        return extra_step_kwargs

    def _rescale_noise_cfg(self, noise_cfg, noise_pred_text, guidance_rescale=0.0):
        """
        Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        return noise_cfg

    def update_unet_ema(self):
        if self.unet_ema_rate is not None:
            for ema_p, cur_p in zip(self.unet_ema.parameters(), self.unet.parameters()):
                ema_p.data = ema_p.data * self.unet_ema_rate + cur_p.data * (1 - self.unet_ema_rate)


@register('diffusion_model_dummy_l2')
class DiffusionModelDummyL2(DiffusionModel):

    def get_losses(self, x, cond=None, input_perturbation=0, prediction_type=None, snr_gamma=None):
        if self.vae is not None:
            x = self.vae.encode(x)['latent_dist'].sample()
            x = x * self.vae.config.scaling_factor

        bsz = x.shape[0]
        # Sample a random timestep for each image
        scheduler = self.train_scheduler
        timesteps = torch.randint(scheduler.config.num_train_timesteps - 1, scheduler.config.num_train_timesteps, (bsz,), device=x.device)
        timesteps = timesteps.long()

        noise = torch.randn_like(x, device=x.device)

        # Predict the noise residual and compute loss
        if cond is None:
            cond = {}
        model_pred = self.unet(noise, timesteps, **cond)['sample']

        target = x
        loss = F.mse_loss(model_pred.float(), target.float(), reduction='mean')
        return loss

    @torch.no_grad()
    def sample(self, batch_size=1, shape=None, num_steps=50, cond=None, cond_neg=None, guidance_scale=1.0, use_unet_ema=False,
               num_images_per_cond=1, eta=0.0, guidance_rescale=0.0, sample=None, generator=None, output_type='image'):
        device = next(self.unet.parameters()).device

        # Prepare timesteps
        scheduler = self.train_scheduler
        timesteps = torch.randint(scheduler.config.num_train_timesteps - 1, scheduler.config.num_train_timesteps, (batch_size,), device=device)
        timesteps = timesteps.long()

        # Prepare sample variables
        if shape is None:
            if isinstance(self.unet, UNet2DModel) or isinstance(self.unet, UNet2DConditionModel):
                shape = (self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size)
            else:
                shape = self.unet.config.sample_shape
        shape = (batch_size,) + tuple(shape)
        assert not (isinstance(generator, list) and len(generator) != batch_size)

        noise = torch.randn(*shape, device=device)

        # Predict the noise residual
        if cond is None:
            cond = {}
        unet = self.unet_ema if use_unet_ema else self.unet
        sample = unet(noise, timesteps, **cond)['sample']

        if self.vae is not None and output_type == 'image':
            ret = self.vae.decode(sample / self.vae.config.scaling_factor)['sample']
        elif output_type == 'latent':
            ret = sample
        else:
            raise NotImplementedError
        return ret