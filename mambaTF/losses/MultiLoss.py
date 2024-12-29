#import torch
import torchaudio
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import auraloss
import torch
from matplotlib import pyplot as plt
import soundfile as sf
import numpy as np
import librosa

class MultiLoss(_Loss):
    """
    Computes the L1 loss between the Mel-spectrograms of the estimated and target signals.
    """

    def __init__(self,k_mel=1, k_mae=0, k_stft=0, k_snr=0, sample_rate=8000, n_fft=1024, hop_length=320, n_mels=1280, normalized = True,reduction='mean', eps=1e-10, stereo=False):
        super().__init__(reduction=reduction)
        self.k_mel = k_mel
        self.k_mae = k_mae
        self.k_stft = k_stft
        self.k_snr = k_snr
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
            if ests.dim() == 2:
                ests = ests.unsqueeze(1)   # [batch, 1, time]
                targets = targets.unsqueeze(1)  # [batch, 1, time]
            
            ests = self.peak_normalize(ests)
            targets = self.peak_normalize(targets)

            #lest_mel =  torch.log(est_mel+self.eps)
            #ltrg_mel =  torch.log(target_mel+self.eps)
            #lest_mel = self.peak_normalize(lest_mel)
            #ltrg_mel = self.peak_normalize(ltrg_mel)

            MAEloss = 0
            MELloss = 0
            STFTloss = 0
            SNRLoss = 0
            
            if self.k_mae > 0:
                MAEloss = self.l1_loss(ests, targets)
            if self.k_stft > 0:
                mrstft = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=[8192, 2048, 512], w_phs=1, w_log_mag=0)
                STFTloss = mrstft(ests, targets)
            if self.k_mel > 0:
                est_mel = self.mel_spectrogram(ests)
                target_mel = self.mel_spectrogram(targets)
                MELloss = nn.functional.l1_loss(est_mel, target_mel, reduction=self.reduction)

            if self.k_snr > 0:
                snr_loss_fn = auraloss.time.SNRLoss()  # Instantiate the SNRLoss class
                SNRLoss = snr_loss_fn(ests, targets)  # Apply it to the inputs

            #print(str(MAEloss)+" "+str(STFTloss)+" "+str(MELloss)+" "+str(SNRLoss))

            return self.k_mel*MELloss + self.k_mae*MAEloss + self.k_stft*STFTloss/8 + self.k_snr*SNRLoss/8
    
    def print_mel(self, input, sr): 
        input = input.astype(np.float32)/ 32768.0
        input = librosa.resample(y=input, target_sr=self.sample_rate, orig_sr=sr)
        #input = librosa.effects.pitch_shift(input, self.sample_rate, n_steps=-12)
        input = torch.from_numpy(input)
        input = self.peak_normalize(input)

        mel = self.mel_spectrogram(input)
        log_mel = torch.log(mel+self.eps)
        #log_mel = self.peak_normalize(log_mel)
        plt.imshow(log_mel, vmin = -4, vmax= 5, origin='lower')
        plt.savefig("/nas/home/gfraticcioli/projects/MambaTransfer/temp/mel.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def peak_normalize(self,input): 
        i_min = input.min()
        i_max = input.max()
        if i_min != i_max: 
            return (input - i_min) / (i_max - i_min)
        else :
            return input

        
if __name__ == "__main__":
    #MMLoss = MelMAESpectrogramLoss(k_mae=1,k_stft=0).cpu()
    #print(MMLoss(torch.rand(1,60000).cpu(), torch.rand(1,60000).cpu()))
    MMLoss =  MultiLoss(k_mel=1,k_stft=0,sample_rate=32000,n_mels=600, n_fft=16384).cpu()
    audio, sr = sf.read("/nas/home/gfraticcioli/projects/MambaTransfer/temp/individualAudio3.wav",start=0, stop=None ,dtype="int16")
    
    MMLoss.print_mel(audio, sr)