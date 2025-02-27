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
is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")
from .base_model import BaseModel
from functools import partial

# from mamba_ssm.modules.mamba_simple import Mamba
# from mamba_ssm.modules.mamba_simple import Block
# from mamba_ssm.models.mixer_seq_simple import _init_weights
# from mamba_ssm.ops.triton.layernorm import RMSNorm #-> torch 2.0.0

#EULER
#from mamba_ssm.modules.mamba2_simple import Mamba2Simple
#from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.block import Block
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.ops.triton.layer_norm import RMSNorm #-> torch 2.0.0

# Modified model without skip connections.
# Modified model: No skip connections and no fusion blocks.
class SnakeNet(nn.Module):
    def __init__(self, dim,n_mambas=1, stride=2, kernel_sizes=[16, 32, 64]):  # Dynamically chosen kernel sizes
        super(SnakeNet, self).__init__()
        # # Encoder
        # self.enc1 = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=dim//4,  # Changed to integer division
        #             kernel_size=kernel_sizes[0], stride=stride,
        #             padding=(kernel_sizes[0] - stride) // 2),  
        #     nn.BatchNorm1d(dim//4),  # Ensure BN layer also gets integer
        #     nn.ReLU()
        # )
        # self.enc2 = nn.Sequential(
        #     nn.Conv1d(in_channels=dim//4, out_channels=dim//2,  # Integer division
        #             kernel_size=kernel_sizes[1], stride=stride,
        #             padding=(kernel_sizes[1] - stride) // 2),
        #     nn.BatchNorm1d(dim//2),  # Adjusted
        #     nn.ReLU()
        # )
        # self.enc3 = nn.Sequential(
        #     nn.Conv1d(in_channels=dim//2, out_channels=dim,  # Already integer
        #             kernel_size=kernel_sizes[2], stride=stride,
        #             padding=(kernel_sizes[2] - stride) // 2),
        #     nn.BatchNorm1d(dim),
        #     nn.ReLU()
        # )

        # # Decoder
        # self.dec1 = nn.Sequential(
        #     nn.ConvTranspose1d(in_channels=2*dim, out_channels=dim//4,
        #                     kernel_size=kernel_sizes[2], stride=stride,
        #                     padding=(kernel_sizes[2] - stride) // 2),
        #     nn.BatchNorm1d(dim//4),
        #     nn.ReLU()
        # )
        # self.dec2 = nn.Sequential(
        #     nn.ConvTranspose1d(in_channels=3*dim//4, out_channels=dim//2,  # Fixed
        #                     kernel_size=kernel_sizes[1], stride=stride,
        #                     padding=(kernel_sizes[1] - stride) // 2),
        #     nn.BatchNorm1d(dim//2),
        #     nn.ReLU()
        # )
        # self.dec3 = nn.ConvTranspose1d(in_channels=3*dim//4, out_channels=1,
        #                             kernel_size=kernel_sizes[0], stride=stride,
        #                             padding=(kernel_sizes[0] - stride) // 2)
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2 * dim,
                    kernel_size=kernel_sizes[0], stride=stride,
                    padding=(kernel_sizes[0] - stride) // 2),  
            nn.BatchNorm1d(2 * dim),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(in_channels=2 * dim, out_channels= dim,
                    kernel_size=kernel_sizes[1], stride=stride,
                    padding=(kernel_sizes[1] - stride) // 2),
            nn.BatchNorm1d(dim),
            nn.ReLU()
        )


        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=3 * dim, out_channels=2 * dim,
                            kernel_size=kernel_sizes[1], stride=stride,
                            padding=(kernel_sizes[1] - stride) // 2),
            nn.BatchNorm1d(2 * dim),
            nn.ReLU()
        )
        self.dec3 = nn.ConvTranspose1d(in_channels=4 * dim, out_channels=1,
                                    kernel_size=kernel_sizes[0], stride=stride,
                                    padding=(kernel_sizes[0] - stride) // 2)


        # The Snake block remains unchanged.

        self.Snake = nn.ModuleList([])
        for _ in range(n_mambas):
            self.Snake.append(
                Block(
                    dim=dim,
                    mixer_cls=partial(Mamba, layer_idx=0, d_state=8, d_conv=2, expand=4),
                    norm_cls=partial(RMSNorm, eps=1e-5),
                    fused_add_norm=False,
                    mlp_cls=nn.Identity
                    )
                )
        self.BSnake = nn.ModuleList([])
        for _ in range(n_mambas):
            self.BSnake.append(
                Block(
                    dim=dim,
                    mixer_cls=partial(Mamba, layer_idx=0, d_state=8, d_conv=2, expand=4),
                    norm_cls=partial(RMSNorm, eps=1e-5),
                    fused_add_norm=False,
                    mlp_cls=nn.Identity
                    )
                )
        self.apply(partial(_init_weights, n_layer=1))


    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, L)
        x = x+1
        # Encoder forward
        x1 = self.enc1(x)   # (B, 1024, L1)
        x2 = self.enc2(x1)  # (B, 2048, L2)
        x3 = x2
        
        # Process the bottleneck features with the Snake block.
        x3 = x3.transpose(1, 2)  # (B, L3, 512)
        forward_f = x3.clone()
        for_residual = None
        for mamba in self.Snake:  # Iterate through Block modules
            forward_f, for_residual = mamba(forward_f,for_residual)  # Apply each Block in sequence
        
        Bforward_f = x3.clone()
        Bfor_residual = None
        Bforward_f = torch.flip(Bforward_f, [1])
        for mamba in self.BSnake:  # Iterate through Block modules
            Bforward_f, Bfor_residual = mamba(Bforward_f,Bfor_residual)  # Apply each Block in sequence

        residual = (forward_f + for_residual) if for_residual is not None else forward_f
        Bresidual = (Bforward_f + Bfor_residual) if Bfor_residual is not None else Bforward_f
        Bresidual = torch.flip(Bresidual, [1])

        residual = torch.cat([residual, Bresidual], -1)
        x3 = residual
        
        x3 = x3.transpose(1, 2)  # (B, 2*512, L3)

        # Decoder forward using transposed convolutions only.
        #d1 = self.dec1(x3)  # Upsample: (B, 1024, ?)
        d1 = x3
        d1 = torch.cat([d1, x2], dim=1)  # (B, 3072, L/16)
        d2 = self.dec2(d1)  # Upsample: (B, 512, ?)
        d2 = torch.cat([d2, x1], dim=1)  # (B, 1536, L/4)
        out = self.dec3(d2) # Final reconstruction: (B, 1, original_length)
        
        out = out - 1 
        return out[:,0,:]




class MambaNet(BaseModel):
    def __init__(
        self,
        dim,
        stride=2,
        sample_rate=48000, 
        n_layers=1,
        n_mambas=1,
        kernel_sizes=[16, 32, 64]
    ):
        super().__init__(sample_rate)
        self.n_layers = n_layers
        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                SnakeNet(
                    dim = dim,
                    n_mambas=n_mambas,
                    stride = stride
                    )
                )

    def forward(
        self,
        input: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
        n_samples = input.shape[1] # [B,T]
        batch = input
        batch = input # [B, M, T]
        for ii in range(self.n_layers):
            batch = self.blocks[ii](batch)  # [B, -1, T]
        batch = self.normalize_batch(batch,input)
        return batch

    def normalize_batch(self, batch, input, eps=1e-8):
        # Compute the RMS for each input element along the feature dimension
        # Assuming the dimensions are (B, C, T, ...)
        input_rms = torch.sqrt(torch.mean(input ** 2, dim=tuple(range(1, input.ndim)), keepdim=True) + eps)
        
        # Compute the RMS for each batch element along the feature dimension
        batch_rms = torch.sqrt(torch.mean(batch ** 2, dim=tuple(range(1, batch.ndim)), keepdim=True) + eps)
        
        # Avoid division by zero by ensuring RMS is greater than eps
        mask = batch_rms > eps
        
        # Normalize batch where RMS is valid
        batch = torch.where(mask, (input_rms / batch_rms) * batch, batch)
        
        # Fallback normalization if batch_rms is not greater than eps
        if not mask.any():
            batch_max = batch.max()
            if batch_max > eps:
                batch = batch / batch_max
        
        return batch
        
    @staticmethod
    def pad2(input_tensor, target_len):
        input_tensor = torch.nn.functional.pad(
            input_tensor, (0, target_len - input_tensor.shape[-1])
        )
        return input_tensor
    
    def get_model_args(self):
        model_args = {"n_sample_rate": 2}
