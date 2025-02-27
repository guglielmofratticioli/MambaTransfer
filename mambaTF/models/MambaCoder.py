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

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.block import Block
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.ops.triton.layer_norm import RMSNorm #-> torch 2.0.0

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )
        
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv. Uses bilinear upsampling by default."""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        else:
            # Use transposed convolution
            self.up = nn.ConvTranspose1d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        # x1 is from the decoder (to be upsampled)
        # x2 is the corresponding skip connection from the encoder
        x1 = self.up(x1)
        # Pad x1 if necessary to match x2's length (handle odd dimensions)
        diff = x2.size(2) - x1.size(2)
        if diff > 0:
            x1 = F.pad(x1, (diff // 2, diff - diff // 2))
        elif diff < 0:
            x2 = F.pad(x2, (-diff // 2, -diff - (-diff) // 2))
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class MambaCBlock(nn.Module):
    def __init__(self,
                 length,
                 dim,
                 hop,
                 swap_DL=True,
                 eps=1e-5,
                 headdim=64,
                 expand=4,
                 n_mamba=1,
                 bidirectional=False):
        super(MambaCBlock, self).__init__()

        self.dim = dim
        self.length = length
        self.hop = hop # 0.0 - 1.0
        self.eps = eps
        self.n_mamba = n_mamba
        self.bidirectional = bidirectional

        self.forward_blocks = nn.ModuleList([])
        for i in range(n_mamba):
            self.forward_blocks.append(
                Block(
                    dim=self.dim,
                    #mixer_cls=partial(Mamba2Simple, layer_idx=i, d_state=16, headdim=headdim, d_conv=4, expand=expand, use_causal_conv1d_fn=False,use_mem_eff_path=False),
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
                        #mixer_cls=partial(Mamba2Simple, layer_idx=i, d_state=16, headdim=headdim, d_conv=4, expand=expand, use_causal_conv1d_fn=False,use_mem_eff_path=False),
                        #mixer_cls=partial(Mamba, layer_idx=i, d_state=16, headdim=headdim, d_conv=4, expand=expand, use_mem_eff_path=True),
                        mixer_cls=partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=expand),
                        norm_cls=partial(RMSNorm, eps=1e-5),
                        fused_add_norm=False,
                        mlp_cls= nn.Identity
                    )
                )

        bilinear=True
        # Encoder: Each "Down" halves the time resolution.
        self.inc = DoubleConv(1, 64)        # [B, 64, 12288] - [B, 64, 4096]
        self.down1 = Down(64, 128)                    # [B, 128, 6144] - [B, 64, 2048]
        self.down2 = Down(128, 256)                   # [B, 256, 3072] - [B, 64, 1024]
        self.down3 = Down(256, 512)                   # [B, 512, 1536] - [B, 64, 512]
        factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)        # [B, 1024//factor, 768] - [B, 64, 256]

        # # Decoder: Upsample step by step and concatena skip connections.
        # self.up1 = Up(1024, 512 // factor, bilinear)    # output: [B, 512//factor, 1536] - [B, 512//factor, 512]

        # Modified bottleneck: from 512 to 4096 channels (length becomes 256)
        self.down4 = Down(512, 4096)                   # [B, 4096, 256]

        # Decoder path
        # For the first Up, the input channels equals the concatenation of x5 (4096) and skip x4 (512) → 4608 channels.
        self.up1 = Up(4608, 512, bilinear)   
        self.up2 = Up(512, 256 , bilinear)     # output: [B, 256//factor, 3072] - [B, 256//factor, 1024]
        self.up3 = Up(256, 128 , bilinear)     # output: [B, 128//factor, 6144] - [B, 128//factor, 2048]
        self.up4 = Up(128, 64, bilinear)                # output: [B, 64, 12288] - [B, 64//factor, 4096]
        self.outc = OutConv(64, 1)           # final output: [B, out_channels, 12288] - [B, out_channels, 512]
    
        '''
        # # Encoder
        # self.encoder = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=2048, kernel_size=8, stride=2, padding=3),  # (B, 2048, 2048)
        #     nn.BatchNorm1d(2048),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=2048, out_channels=4096, kernel_size=8, stride=2, padding=3),  # (B, 4096, 1024)
        #     nn.BatchNorm1d(4096),
        #     nn.ReLU(),
        # )

        # # Decoder
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose1d(in_channels=4096, out_channels=2048, kernel_size=8, stride=2, padding=3, output_padding=0),  # (B, 2048, 2048)
        #     nn.BatchNorm1d(2048),
        #     nn.ReLU(),
        #     nn.ConvTranspose1d(in_channels=2048, out_channels=1, kernel_size=8, stride=2, padding=3, output_padding=0),  # (B, 1, 4096)
        # )


        # # Single-layer Encoder (Downsampling)
        # self.encoder = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=4096, kernel_size=1)
        # )

        # # Single-layer Decoder (Upsampling)
        # self.decoder = nn.Sequential(
        #     nn.Conv1d(in_channels=4096, out_channels=1, kernel_size=1)
        # )
                
        # self.encoder = nn.Sequential(
        #     nn.Conv1d(in_channels=1, out_channels=1024, kernel_size=8, stride=4, padding=2),  # (B, 1024, 3072)
        #     nn.BatchNorm1d(1024),  # Normalize over channels
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=1024, out_channels=2048, kernel_size=8, stride=4, padding=2),  # (B, 2048, 768)
        #     nn.BatchNorm1d(2048),  # Normalize over channels
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=2048, out_channels=4096, kernel_size=8, stride=2, padding=2),  # (B, 4096, 384)
        #     nn.BatchNorm1d(4096),  # Normalize over channels
        #     nn.ReLU()
        # )

        # # Decoder
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose1d(in_channels=4096, out_channels=2048, kernel_size=8, stride=2, padding=2, output_padding=0),  # (B, 2048, 768)
        #     nn.BatchNorm1d(2048),
        #     nn.ReLU(),
        #     nn.ConvTranspose1d(in_channels=2048, out_channels=1024, kernel_size=8, stride=4, padding=2, output_padding=0),  # (B, 1024, 3072)
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        #     nn.ConvTranspose1d(in_channels=1024, out_channels=1, kernel_size=8, stride=4, padding=2, output_padding=0),  # (B, 1, 12288)
        #     #nn.Tanh()  # Output between -1 and 1
        #     #nn.ReLU()
        # )

        # assert self.d_inner % self.headdim == 0
        # self.d_inner = self.expand * self.d_model
        # causal conv 1d  stride rule -> -> d_model * expand / headdim = multiple of 8
        '''
        
        self.apply(partial(_init_weights, n_layer=n_mamba))

    def forward(self, x):

        # intra RNN
        # [B,T]
        input_ = x
        B, T = input_.shape
        batch = input_.unsqueeze(1)  # [B, 1, T]
        
        batch = batch + 1 # DC OFFSET
        #batch = self.encoder(batch)  # [B, C, T]

        x1 = self.inc(batch)       # [B, 64, 12288]
        x2 = self.down1(x1)    # [B, 128, 6144]
        x3 = self.down2(x2)    # [B, 256, 3072]
        x4 = self.down3(x3)    # [B, 512, 1536]
        x5 = self.down4(x4)    # [B, 1024//factor, 768]

        batch = x5
        # Expects input [B, T, C] where C = embeddings channels
        batch = batch.transpose(1,2)  # [B, T, C]
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
            #residual = torch.cat([residual, back_residual], -1)
            #residual = residual.transpose(1, 2)  # [B, H, -1]mambaTF/models/JustMamba2.py
        
        residual = residual.transpose(1, 2)  # [B, T, H]
        
        x5 = residual 
        x = self.up1(x5, x4)   # [B, 512//factor, 1536]
        x = self.up2(x, x3)    # [B, 256//factor, 3072]
        x = self.up3(x, x2)    # [B, 128//factor, 6144]
        x = self.up4(x, x1)    # [B, 64, 12288]
        logits = self.outc(x)  # [B, out_channels, 12288]

        #residual = self.decoder(residual)  # [B, 1, T]

        logits = logits - 1 # DC OFFSET

        out = logits[:,0,:]
        #out = residual #+ input_  # [B, C, T]

        return out
        #

class MambaCoder(BaseModel):
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
        eps=1.0e-5,
        bidirectional = False,
        sample_rate=32000
    ):
        super().__init__(sample_rate)
        self.n_layers = n_layers
        self.n_mamba = n_mamba
        self.bidirectional = bidirectional

        self.eps = eps
        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                MambaCBlock(
                    dim = dim,
                    length = length,
                    hop = hop,
                    headdim=headdim,
                    expand=expand,
                    eps = eps,
                    n_mamba=self.n_mamba,
                    swap_DL=swap_DL,
                    bidirectional=self.bidirectional),
                )

    def forward(
        self,
        input: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:


        n_samples = input.shape[1] # [B,T]
        batch = input

        # EMBEDDING CONVOLUTION
        #batch = batch.unsqueeze(1)
        #batch = self.conv_block(batch)  # [B, n_chan, T]
        #batch = batch.transpose(1,2)

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
    
    def peak_normalize(batch: torch.Tensor, target_peak: float = 1.0) -> torch.Tensor:
        """
        Peak normalizes a batch of audio tensors.

        Args:
            batch (torch.Tensor): Batch of audio tensors of shape (B, C, T) or (B, T), 
                                where B is batch size, C is number of channels, T is the number of samples.
            target_peak (float): Desired peak value (e.g., 1.0 or -1.0).

        Returns:
            torch.Tensor: Peak-normalized batch of audio tensors.
        """
        # Ensure target_peak is positive
        target_peak = abs(target_peak)

        # Compute the peak (max absolute value) for each audio sample
        max_vals = batch.abs().flatten(start_dim=1).max(dim=1).values  # Shape: (B,)

        # Avoid division by zero by adding a small epsilon
        max_vals = max_vals.clamp(min=1e-8)

        # Reshape for broadcasting (B, 1, 1) if 3D, (B, 1) if 2D
        if batch.dim() == 3:
            max_vals = max_vals[:, None, None]
        else:
            max_vals = max_vals[:, None]

        # Scale the batch to the target peak
        normalized_batch = (batch / max_vals) * target_peak

        return normalized_batch

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