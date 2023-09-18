import torch

import utils
from trainers import register
from .base_trainer import BaseTrainer


def pixel_values_to_image(pixel_values):
    return (pixel_values * 0.5 + 0.5).clamp(0, 1)


@register('dae_trainer')
class DAETrainer(BaseTrainer):

    def make_model(self, model_spec=None):
        super().make_model(model_spec)
        m = self.model
        self.log(f'  encoder: {utils.compute_num_params(m.encoder)}')
        self.log(f'  dm.unet: {utils.compute_num_params(m.dm.unet)}')
        self.log(f'  dm.vae.encoder: {utils.compute_num_params(m.dm.vae.encoder)}')
        self.log(f'  dm.vae.decoder: {utils.compute_num_params(m.dm.vae.decoder)}')

    def train_step(self, data, bp=True):
        ret = super().train_step(data, bp=bp)
        self.model.dm.update_unet_ema()
        return ret

    def visualize(self):
        self.model_ddp.eval()
        with torch.no_grad():
            self.visualize_dae_samples('train', self.datasets['train'].vis_samples)
            self.visualize_dae_samples('val', self.datasets['val'].vis_samples)

    def visualize_dae_samples(self, name, data_list):
        if self.is_master:
            for i, data in enumerate(data_list):
                if i == self.cfg.visualize.n_vis:
                    break
                self.log_image(f'{name}/{i}_image_prompt', pixel_values_to_image(data['encoder_pixel_values']))

                data = {k: v.cuda() for k, v in data.items()}
                z = self.model.encoder(data['encoder_pixel_values'].unsqueeze(0))
                if self.model.condition_neck is not None:
                    z = self.model.condition_neck(z)
                shape = self.cfg.visualize.get('sample_shape', None)

                sample_pixel_values = self.model.dm.sample(cond=z, shape=shape)[0]
                self.log_image(f'{name}/{i}_generated', pixel_values_to_image(sample_pixel_values))

                if self.model.dm.unet_ema_rate is not None:
                    sample_pixel_values = self.model.dm.sample(cond=z, shape=shape, use_unet_ema=True)[0]
                    self.log_image(f'{name}/{i}_generated_ema', pixel_values_to_image(sample_pixel_values))
