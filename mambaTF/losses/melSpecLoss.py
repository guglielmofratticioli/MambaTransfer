#import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

class MelSpectrogramLoss(_Loss):
    """
    Computes the L1 loss between the Mel-spectrograms of the estimated and target signals.
    """

    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80, reduction='mean'):
        super().__init__(reduction=reduction)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.reduction = reduction
        self.mel_spectrogram = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels
            )
        )

    def forward(self, ests, targets):
        if ests.size() != targets.size() or ests.ndim != 2:
            raise TypeError(f'Inputs must be of shape [batch, time], got {ests.size()} and {targets.size()}')

        ests = ests.unsqueeze(1)  # [batch, 1, time]
        targets = targets.unsqueeze(1)  # [batch, 1, time]

        est_mel = self.mel_spectrogram(ests)
        target_mel = self.mel_spectrogram(targets)

        loss = nn.functional.l1_loss(est_mel, target_mel, reduction=self.reduction)
        return loss