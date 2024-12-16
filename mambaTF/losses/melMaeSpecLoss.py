#import torch
import torchaudio
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import auraloss

class MelMAESpectrogramLoss(_Loss):
    """
    Computes the L1 loss between the Mel-spectrograms of the estimated and target signals.
    """

    def __init__(self,k_mel=0.8, k_mae=0, k_stft=0.2, sample_rate=8000, n_fft=1024, hop_length=320, n_mels=1280, reduction='mean', eps=1e-10, stereo=False):
        super().__init__(reduction=reduction)
        self.k_mel = k_mel
        self.k_mae = k_mae
        self.k_stft = k_stft
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
        self.l1_loss = nn.L1Loss(reduction=reduction)

    def forward(self, ests, targets):
        if ests.size() != targets.size():
            raise TypeError(f'Inputs must be of the same shape, got {ests.size()} and {targets.size()}')
        MAEloss = self.l1_loss(ests, targets)
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

            return self.alpha*(lossL+lossR)/2 + (1-self.alpha)*MAEloss
        else:
            ests = ests.unsqueeze(1)   # [batch, 1, time]
            targets = targets.unsqueeze(1)  # [batch, 1, time]

            est_mel = self.mel_spectrogram(ests)
            target_mel = self.mel_spectrogram(targets)


            MELloss = nn.functional.l1_loss(est_mel, target_mel, reduction=self.reduction)

            mrstft = auraloss.freq.MultiResolutionSTFTLoss( fft_sizes=[8192 ,2048 , 512], w_phs = 1, w_log_mag =0)
            STFTloss = mrstft(ests, targets)

            return self.k_mel*MELloss + self.k_mae*MAEloss + self.k_stft*STFTloss