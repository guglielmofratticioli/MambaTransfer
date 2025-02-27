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
    def __init__(self, dim, stride=2, kernel_sizes=[16, 32, 64]):  # Dynamically chosen kernel sizes
        super(SnakeNet, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4 * dim,
                    kernel_size=kernel_sizes[0], stride=stride,
                    padding=(kernel_sizes[0] - stride) // 2),  
            nn.BatchNorm1d(4 * dim),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(in_channels=4 * dim, out_channels=2 * dim,
                    kernel_size=kernel_sizes[1], stride=stride,
                    padding=(kernel_sizes[1] - stride) // 2),
            nn.BatchNorm1d(2 * dim),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(in_channels=2 * dim, out_channels=dim,
                    kernel_size=kernel_sizes[2], stride=stride,
                    padding=(kernel_sizes[2] - stride) // 2),
            nn.BatchNorm1d(dim),
            nn.ReLU()
        )

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=dim, out_channels=2 * dim,
                            kernel_size=kernel_sizes[2], stride=stride,
                            padding=(kernel_sizes[2] - stride) // 2),
            nn.BatchNorm1d(2 * dim),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=4 * dim, out_channels=2 * dim,
                            kernel_size=kernel_sizes[1], stride=stride,
                            padding=(kernel_sizes[1] - stride) // 2),
            nn.BatchNorm1d(2 * dim),
            nn.ReLU()
        )
        self.dec3 = nn.ConvTranspose1d(in_channels=6 * dim, out_channels=1,
                                    kernel_size=kernel_sizes[0], stride=stride,
                                    padding=(kernel_sizes[0] - stride) // 2)


        # The Snake block remains unchanged.
        self.Snake = Block(
            dim=dim,
            mixer_cls=partial(Mamba, layer_idx=0, d_state=8, d_conv=2, expand=4),
            norm_cls=partial(RMSNorm, eps=1e-5),
            fused_add_norm=False,
            mlp_cls=nn.Identity
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, L)
        x = x+1
        # Encoder forward
        x1 = self.enc1(x)   # (B, 1024, L1)
        x2 = self.enc2(x1)  # (B, 2048, L2)
        x3 = self.enc3(x2)  # (B, 512, L3)
        
        # Process the bottleneck features with the Snake block.
        x3 = x3.transpose(1, 2)  # (B, L3, 512)
        forward_f = x3.clone()
        forward_f, for_residual = self.Snake(forward_f, None)
        x3 = (forward_f + for_residual) if for_residual is not None else forward_f
        x3 = x3 + forward_f
        x3 = x3.transpose(1, 2)  # (B, 512, L3)

        # Decoder forward using transposed convolutions only.
        d1 = self.dec1(x3)  # Upsample: (B, 1024, ?)
        d1 = torch.cat([d1, x2], dim=1)  # (B, 3072, L/16)
        d2 = self.dec2(d1)  # Upsample: (B, 512, ?)
        d2 = torch.cat([d2, x1], dim=1)  # (B, 1536, L/4)
        out = self.dec3(d2) # Final reconstruction: (B, 1, original_length)
        
        out = out - 1 
        return out