import torch
import torch.nn as nn
import torch.nn.functional as F

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


class MaUNet(nn.Module):
    def __init__(self,mamba_dim=1024, in_channels=1, out_channels=1, bilinear=False):
        """
        Args:
            in_channels: number of channels in the input (e.g., 1 for mono audio)
            out_channels: number of channels in the output (e.g., 1 for denoised signal)
            bilinear: whether to use bilinear upsampling or transposed convolution
        """
        super(AudioUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.mamba_dim = mamba_dim
        # Encoder: Each "Down" halves the time resolution.
        self.inc = DoubleConv(in_channels, 64)          # [B, 64, 12288] - [B, 64, 4096]
        self.down1 = Down(64, 128)                      # [B, 128, 6144] - [B, 64, 2048]
        self.down2 = Down(128, 256)                     # [B, 256, 3072] - [B, 64, 1024]
        self.down3 = Down(256, 512)                     # [B, 512, 1536] - [B, 64, 512]
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)          # [B, 1024//factor, 768] - [B, 64, 256]

        self.Mamba = Block(
                    dim=self.mamba_dim,
                    #mixer_cls=partial(Mamba2Simple, layer_idx=i, d_state=16, headdim=headdim, d_conv=4, expand=expand, use_causal_conv1d_fn=False,use_mem_eff_path=False),
                    #mixer_cls=partial(Mamba, layer_idx=i, d_state=16, headdim=headdim, d_conv=4, expand=expand, use_mem_eff_path=True),
                    mixer_cls=partial(Mamba, layer_idx=i, d_state=16, d_conv=4, expand=expand),
                    norm_cls=partial(RMSNorm, eps=1e-5),
                    fused_add_norm=False,
                    mlp_cls= nn.Identity
                )

        # Decoder: Upsample step by step and concatenate skip connections.
        self.up1 = Up(1024, 512 // factor, bilinear)      # output: [B, 512//factor, 1536] - [B, 512//factor, 512]
        self.up2 = Up(512, 256 // factor, bilinear)       # output: [B, 256//factor, 3072] - [B, 256//factor, 1024]
        self.up3 = Up(256, 128 // factor, bilinear)       # output: [B, 128//factor, 6144] - [B, 128//factor, 2048]
        self.up4 = Up(128, 64, bilinear)                  # output: [B, 64, 12288] - [B, 64//factor, 4096]
        self.outc = OutConv(64, out_channels)             # final output: [B, out_channels, 12288] - [B, out_channels, 512]

    def forward(self, x):
        # x: [B, 1, 12288]
        x1 = self.inc(x)       # [B, 64, 12288]
        x2 = self.down1(x1)    # [B, 128, 6144]
        x3 = self.down2(x2)    # [B, 256, 3072]
        x4 = self.down3(x3)    # [B, 512, 1536]
        x5 = self.down4(x4)    # [B, 1024//factor, 768]
        x = self.up1(x5, x4)   # [B, 512//factor, 1536]
        x = self.up2(x, x3)    # [B, 256//factor, 3072]
        x = self.up3(x, x2)    # [B, 128//factor, 6144]
        x = self.up4(x, x1)    # [B, 64, 12288]
        logits = self.outc(x)  # [B, out_channels, 12288]
        return logits