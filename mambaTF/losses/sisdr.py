import torch
#import torch.nn as nn
from torch.nn.modules.loss import _Loss

class SingleSrcNegSDR(_Loss):
    """
    Computes negative SI-SDR loss for single-source signals.
    """

    def __init__(self, sdr_type='sisdr', zero_mean=True, take_log=True, reduction='mean', EPS=1e-8):
        super().__init__(reduction=reduction)
        assert sdr_type in ['snr', 'sisdr', 'sdsdr']
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = EPS

    def forward(self, ests, targets):
        if ests.size() != targets.size() or ests.ndim != 2:
            raise TypeError(f'Inputs must be of shape [batch, time], got {ests.size()} and {targets.size()}')

        if self.zero_mean:
            ests = ests - torch.mean(ests, dim=1, keepdim=True)
            targets = targets - torch.mean(targets, dim=1, keepdim=True)

        dot = torch.sum(ests * targets, dim=1, keepdim=True)
        target_energy = torch.sum(targets ** 2, dim=1, keepdim=True) + self.EPS
        scaled_targets = dot * targets / target_energy

        if self.sdr_type in ['sdsdr', 'snr']:
            e_noise = ests - targets
        else:
            e_noise = ests - scaled_targets

        losses = torch.sum(scaled_targets ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + self.EPS)
        if self.take_log:
            losses = 10 * torch.log10(losses + self.EPS)

        if self.reduction == 'mean':
            return -torch.mean(losses)
        elif self.reduction == 'sum':
            return -torch.sum(losses)
        else:
            return -losses
