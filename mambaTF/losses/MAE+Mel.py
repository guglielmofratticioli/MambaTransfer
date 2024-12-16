
import torch
from torch.nn.modules.loss import _Loss


class MAEMel(_Loss):
    def __init__(self, win=2048, stride=512, alpha=0.5):
        super().__init__()
        self.EPS = 1e-8
        self.win = win
        self.stride = stride
        self.alpha = alpha

    def forward(self, ests, targets):
        B, nsrc, _ = ests.shape
        est_spec = torch.stft(ests.view(-1, ests.shape[-1]), n_fft=self.win, hop_length=self.stride,
                              window=torch.hann_window(self.win).to(ests.device).float(),
                              return_complex=True)
        est_target = torch.stft(targets.view(-1, targets.shape[-1]), n_fft=self.win, hop_length=self.stride,
                                window=torch.hann_window(self.win).to(ests.device).float(),
                                return_complex=True)
        freq_L1 = (est_spec.real - est_target.real).abs().mean((1, 2)) + (est_spec.imag - est_target.imag).abs().mean(
            (1, 2))
        freq_L1 = freq_L1.view(B, nsrc).mean(-1)

        wave_l1 = (ests - targets).abs().mean(-1)
        # print(freq_L1.shape, wave_l1.shape)
        wave_l1 = wave_l1.view(B, nsrc).mean(-1)
        return self.alpha*freq_L1 + (1-self.alpha)*wave_l1