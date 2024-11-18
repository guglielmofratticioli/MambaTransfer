import MAE
import melSpecLoss
import torch
import torch.nn as nn
import torchaudio
from torch.nn.modules.loss import _Loss


class MAEMel(_Loss):
    def __init__(self,alpha, sample_rate=8000, n_fft=1024, hop_length=256, n_mels=80, reduction='mean',
                 stereo=False):
        super().__init__(reduction=reduction)
        self.alpha = alpha
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.reduction = reduction
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
        MAELoss = self.l1_loss(ests, targets)
        est_mel = self.mel_spectrogram(ests)
        target_mel = self.mel_spectrogram(targets)
        MelLoss = nn.functional.l1_loss(est_mel, target_mel, reduction=self.reduction)
        MelLoss = MelLoss*0.01 #Normalization

        loss = self.alpha*(MAELoss) + (1-self.alpha)*MelLoss
        return loss