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
class SnakeNet1D(nn.Module):
    def __init__(self, dim, stride=2, kernel_sizes=[16, 32, 64]):  # Dynamically chosen kernel sizes
        super(SnakeNet1D, self).__init__()

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
    
class MambaBlock(nn.Module):
    def __init__(self,
                 length,
                 dim,
                 hop,
                 swap_DL=True,
                 eps=1e-5,
                 headdim=64,
                 expand=4,
                 n_layer=1,
                 n_mamba=1,
                 snake_dim=1024,
                 stride=1,
                 bidirectional=False,
                 use_SnakeNet=True):
        super(MambaBlock, self).__init__()

        self.dim = dim
        self.length = length
        self.hop = hop # 0.0 - 1.0
        self.eps = eps
        self.n_layer = n_layer
        self.bidirectional = bidirectional
        self.swap_DL = swap_DL
        self.use_SnakeNet = use_SnakeNet

        if use_SnakeNet is True:
            self.snakeNet = SnakeNet1D(dim=snake_dim,stride=stride)

        # assert self.d_inner % self.headdim == 0
        # self.d_inner = self.expand * self.d_model
        # causal conv 1d  stride rule -> -> d_model * expand / headdim = multiple of 8
        self.forward_blocks = nn.ModuleList([])
        for i in range(n_mamba):
            self.forward_blocks.append(
                Block(
                    dim=self.dim,
                    #mixer_cls=partial(Mamba, layer_idx=i, d_state=16, headdim=headdim, d_conv=4, expand=expand, use_mem_eff_path=True),
                    mixer_cls=partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=expand),
                    norm_cls=partial(RMSNorm, eps=1e-5),
                    fused_add_norm=False,
                    mlp_cls= nn.Identity
                )
            )
        self.backward_blocks = None
        if bidirectional:
            self.backward_blocks = nn.ModuleList([])
            for i in range(n_mamba):
                self.backward_blocks.append(
                        Block(
                        dim=self.dim,
                        #mixer_cls=partial(Mamba, layer_idx=i, d_state=16, headdim=headdim, d_conv=4, expand=expand, use_mem_eff_path=True),
                        mixer_cls=partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=expand),
                        norm_cls=partial(RMSNorm, eps=1e-5),
                        fused_add_norm=False,
                        mlp_cls= nn.Identity
                    )
                )

        self.apply(partial(_init_weights, n_layer=n_layer))

    def forward(self, x):

        # intra RNN
        # [B,T]
        input_ = x
        B, T = input_.shape
        batch = input_
        
        # in position 1 put n windows of size dim
        batch = batch.unfold(1, self.dim, int(self.dim*self.hop)) # [B, n, T]

        #batch = self.norm(input_)  # [B, C, T]
        # Expects input [B, T, C] where C = embeddings channels
        for_residual = None
        forward_f = batch.clone()
        for block in self.forward_blocks:
            forward_f, for_residual = block(forward_f, for_residual)
        residual = (forward_f + for_residual) if for_residual is not None else forward_f

        if self.bidirectional:
            back_residual = None
            backward_f = torch.flip(batch, [1])
            for block in self.backward_blocks:
                backward_f, back_residual = block(backward_f, back_residual)
            back_residual = (backward_f + back_residual) if back_residual is not None else backward_f

            back_residual = torch.flip(back_residual, [1])
            residual = (residual + back_residual) / 2  # Media delle due direzioni

        #residual = residual.view([B, C, T])
        #residual = residual.flatten(start_dim=1)
        # Step 1: Create a triangular window of size 4096
        if self.hop < 1:
            triangular_window = torch.linspace(0, 1, int(self.dim/2))  # First half of the triangle
            triangular_window = torch.cat((triangular_window, torch.linspace(1, 0, int(self.dim/2)))) # Second half of the triangle
            triangular_window = triangular_window.unsqueeze(0).unsqueeze(0)  # [1, 1, 4096]
            triangular_window = triangular_window.expand(residual.size(0), residual.size(1), -1).to(residual.device)  # [1, 15, 4096]
            residual = residual * triangular_window
        
        residual = F.fold(
            residual.permute(0,2,1),
            output_size=(1, T),
            kernel_size=(1, self.dim),
            stride=(1, int(self.dim*self.hop))
            )
        residual = residual[:,0,0,:]
        
        x = residual + input_

        if self.use_SnakeNet: 
            # Bottleneck to reduce tremolo effect
            #x = self.encoder(x)
            x = self.snakeNet(x)
            #x = self.decoder(x)
            x = x[:,0,:]  # [B, 1, T] -> [B, T]

        out = x

        return out

class MambaMix_Stride(BaseModel):
    def __init__(
        self,
        dim,
        length,
        hop,
        headdim=64,
        expand=2,
        swap_DL=True,
        n_layers=1,
        n_mamba=1,
        snake_dim=1024,
        stride=1,
        eps=1.0e-5,
        bidirectional = False,
        use_SnakeNet=True,
        sample_rate=32000
    ):
        super().__init__(sample_rate)
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.eps = eps
        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                MambaBlock(
                    dim = dim,
                    length = length,
                    hop = hop,
                    headdim=headdim,
                    expand=expand,
                    eps = eps,
                    n_layer=1,
                    n_mamba=n_mamba,
                    snake_dim=snake_dim,
                    stride=1,
                    swap_DL=swap_DL,
                    bidirectional=self.bidirectional,
                    use_SnakeNet=use_SnakeNet)
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
        input_rms = torch.sqrt(torch.mean(input ** 2, dim=tuple(range(1, input.ndim)), keepdim=True))
        
        # Compute the RMS for each batch element along the feature dimension
        batch_rms = torch.sqrt(torch.mean(batch ** 2, dim=tuple(range(1, batch.ndim)), keepdim=True))
        
        # Avoid division by zero by ensuring RMS is greater than eps
        mask = batch_rms > eps
        
        # Normalize batch where RMS is valid
        if input_rms > eps and batch_rms > eps: 
            batch = torch.where(mask, (input_rms / batch_rms) * batch, batch)
        elif batch.max() > eps: 
            batch = batch/batch.max()
            
        return batch
    
    @staticmethod
    def pad2(input_tensor, target_len):
        input_tensor = torch.nn.functional.pad(
            input_tensor, (0, target_len - input_tensor.shape[-1])
        )
        return input_tensor
    
    def get_model_args(self):
        model_args = {"n_sample_rate": 2}

