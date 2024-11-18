#import torch
import torchaudio
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

class MelSpectrogramLoss(_Loss):
    """
    Computes the L1 loss between the Mel-spectrograms of the estimated and target signals.
    """

    def __init__(self, sample_rate=8000, n_fft=1024, hop_length=256, n_mels=80, reduction='mean', eps=1e-10, stereo=False):
        super().__init__(reduction=reduction)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.reduction = reduction
        self.eps = eps  # Small constant to prevent log(0)
        self.stereo = stereo

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mel_spectrogram = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                power=1.0  # Use magnitude instead of power
            ).to(device)
        )

    def forward(self, ests, targets):
        if ests.size() != targets.size():
            raise TypeError(f'Inputs must be of the same shape, got {ests.size()} and {targets.size()}')
        if self.stereo:
            estsL = ests[0,:,0]
            estsR = ests[0, :, 1]
            targetsL = targets[0,:,0]
            targetsR = targets[0, :, 1]

            estsL_mel = self.mel_spectrogram(estsL)
            estsR_mel = self.mel_spectrogram(estsR)
            targetsL_mel = self.mel_spectrogram(targetsL)
            targetsR_mel = self.mel_spectrogram(targetsR)

            lossL = nn.functional.l1_loss(estsL_mel, targetsL_mel)
            lossR = nn.functional.l1_loss(estsR_mel, targetsR_mel)

            return (lossL+lossR)/2
        else:
            ests = ests.unsqueeze(1)   # [batch, 1, time]
            targets = targets.unsqueeze(1)  # [batch, 1, time]

            est_mel = self.mel_spectrogram(ests)
            target_mel = self.mel_spectrogram(targets)


            loss = nn.functional.l1_loss(est_mel, target_mel, reduction=self.reduction)
            return loss