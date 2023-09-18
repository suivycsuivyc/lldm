import copy
import math

import torch
import torch.nn as nn

import models
from models import register


@register('empty_text_modifier')
class EmptyTextModifier(nn.Module):

    def __init__(self, ckpt, input_l, input_d):
        super().__init__()
        self.emb = nn.Parameter(torch.load(ckpt, map_location='cpu'))
        assert self.emb.shape[0] == 1 and self.emb.dim() == 3
        self.matl = nn.Parameter(torch.zeros(self.emb.shape[1], input_l))
        matr_t = torch.zeros(self.emb.shape[2], input_d)
        nn.init.kaiming_uniform_(matr_t, a=math.sqrt(5))
        self.matr = nn.Parameter(matr_t.t().detach())

    def forward(self, z):
        x = z['decoding_features']
        matl = self.matl.expand(x.shape[0], -1, -1)
        matr = self.matr.expand(x.shape[0], -1, -1)
        ret = self.emb + torch.bmm(torch.bmm(matl, x), matr)
        return {'decoding_features': ret}


@register('dae')
class DAE(nn.Module):

    def __init__(self, encoder, dm, condition_neck=None):
        super().__init__()
        self.encoder = models.make(encoder)
        self.dm = models.make(dm)
        if condition_neck is not None:
            condition_neck = copy.deepcopy(condition_neck)
            condition_neck['args']['input_l'] = max(self.encoder.config.num_decoding_tokens, 1)
            condition_neck['args']['input_d'] = self.encoder.config.hidden_size
            self.condition_neck = models.make(condition_neck)
        else:
            self.condition_neck = None

    def forward(self, data):
        z = self.encoder(data['encoder_pixel_values'])
        if self.condition_neck is not None:
            z = self.condition_neck(z)
        ret = self.dm.get_losses(data['dm_pixel_values'], cond=z)
        return {'loss': ret}
