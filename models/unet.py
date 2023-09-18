from diffusers import UNet2DConditionModel

from models import register


class UNet2DConditionModelWrapped(UNet2DConditionModel):

    def forward(self, sample, timestep, decoding_features, patch_features=None):
        return super().forward(sample, timestep, encoder_hidden_states=decoding_features)
