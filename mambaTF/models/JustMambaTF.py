import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
#import difflib
import torch
import torch.nn as nn
import torch.nn.functional as F
from click.core import batch
from torch.nn import init
from torch.nn.parameter import Parameter

from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor
is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")

from ..layers import Stft
from ..utils.complex_utils import is_torch_complex_tensor
from ..utils.complex_utils import new_complex_like
from ..utils.get_layer_from_string import get_layer
from .base_model import BaseModel

from functools import partial
from mamba_ssm.modules.mamba_simple import Mamba, Block
#from mamba_ssm.modules.block import Block #Euler
from mamba_ssm.models.mixer_seq_simple import _init_weights
#from mamba_ssm.ops.triton.layer_norm import RMSNorm #Euler
from mamba_ssm.ops.triton.layernorm import RMSNorm #-> torch 2.0.0

class MambaBlock(nn.Module):
    def __init__(self,
                 emb_dim,
                 emb_ks,
                 eps=1e-5,
                 n_layer=1,
                 bidirectional=False):
        super(MambaBlock, self).__init__()

        self.in_channels = emb_dim * emb_ks
        # in_channels = 1

        self.norm = LayerNormalization4D(emb_dim, eps=eps)
        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.eps = eps
        self.n_layer = n_layer

        self.linear = nn.ConvTranspose1d(
            self.in_channels * 2, emb_dim, emb_ks
        )

        self.forward_blocks = nn.ModuleList([])
        for i in range(n_layer):
            self.forward_blocks.append(
                Block(
                    dim=self.in_channels,
                    mixer_cls=partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=4),
                    norm_cls=partial(RMSNorm, eps=1e-5),
                    fused_add_norm=False,
                )
            )
        self.backward_blocks = None
        if bidirectional:
            self.backward_blocks = nn.ModuleList([])
            for i in range(n_layer):
                self.backward_blocks.append(
                        Block(
                        dim=self.in_channels,
                        mixer_cls=partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=4),
                        norm_cls=partial(RMSNorm, eps=1e-5),
                        fused_add_norm=False,
                    )
                )

        self.apply(partial(_init_weights, n_layer=n_layer))

    def forward(self, x):

        # C -> emb_dim convolutional maps
        #B, C, old_T = x.shape
        #T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        #x = F.pad(x, (0, 0, T - old_T))

        # intra RNN
        input_ = x
        batch = input_
        batch = batch.unsqueeze(1)

        #batch = self.norm(input_)  # [B, C, T]

        batch = F.unfold(
            batch, (self.emb_ks, 1)
        )  # [B, C*emb_ks, -1]

        batch = batch.transpose(1, 2)  # [B, -1, C*emb_ks]
        #

        # Expects input [B, T, C*emb_ks] where C = emb_dim -> channel filters

        for_residual = None
        forward_f = batch.clone()
        for block in self.forward_blocks:
            forward_f, for_residual = block(forward_f, for_residual, inference_params=None)
        residual = (forward_f + for_residual) if for_residual is not None else forward_f

        if self.backward_blocks is not None:
            back_residual = None
            backward_f = torch.flip(batch, [1])
            for block in self.backward_blocks:
                backward_f, back_residual = block(backward_f, back_residual, inference_params=None)
            back_residual = (backward_f + back_residual) if back_residual is not None else backward_f

            back_residual = torch.flip(back_residual, [1])
            residual = torch.cat([residual, back_residual], -1)

        #return residual
        #
        residual = residual.transpose(1, 2)  # [B, H, -1]
        residual = self.linear(residual)  # [B, C, T]
        #residual = residual.view([B, C, T])
        residual = residual.transpose(1, 2)

        out = residual + input_  # [B, C, T]

        return out
        #

class JustMambaTF(BaseModel):
    def __init__(
        self,
        n_chan=2,
        n_layers=6,
        emb_dim=1,
        emb_ks=4,
        eps=1.0e-5,
        sample_rate=32000
    ):
        super().__init__(sample_rate)
        self.n_layers = n_layers
        self.n_chan = n_chan

        self.eps = eps

        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                MambaBlock(
                    emb_dim = emb_dim,
                    emb_ks = emb_ks,
                    eps = eps,
                    n_layer=1,
                    bidirectional=True)
                )

    def forward(
        self,
        input: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
        # input shape: (B, T, M)
        was_one_d = False
        if input.ndim == 1:
            was_one_d = True
            input = input.unsqueeze(0).unsqueeze(2)
        elif input.ndim == 2:
            was_one_d = True
            input = input.unsqueeze(2)
        elif input.ndim == 3:
            pass

        n_samples = input.shape[1] #T

        batch = input # [B, M, T]
        for ii in range(self.n_layers):
            batch = self.blocks[ii](batch)  # [B, -1, T]

        # Ensure the output has the correct length
        #batch = self.pad2(batch, n_samples)  # [B, 2, N_samples]

        input_rms = torch.sqrt(torch.mean(input[:,:,0] ** 2, dim=1, keepdim=True))
        batch_rms = torch.sqrt(torch.mean(batch ** 2, dim=1, keepdim=True)) # Shape: (B, 1)

        if batch_rms > self.eps :
            batch = input_rms/batch_rms * batch

        return batch[:,:,0]

    @staticmethod
    def pad2(input_tensor, target_len):
        input_tensor = torch.nn.functional.pad(
            input_tensor, (0, target_len - input_tensor.shape[-1])
        )
        return input_tensor
    
    def get_model_args(self):
        model_args = {"n_sample_rate": 2}


class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            _, C, _, _ = x.shape
            stat_dim = (1,)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat

class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat