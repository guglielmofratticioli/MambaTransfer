import torch
import torch.nn.functional as F
import torchaudio
from matplotlib import pyplot as plt

def rms_envelope(audio, frame_size=1024, hop_size=512, eps=1e-8):
    """
    Extract the RMS envelope of an audio signal.
    
    Args:
        audio (torch.Tensor): Input audio signal, shape [B, T] where B is batch size, T is time.
        frame_size (int): Size of each frame for RMS computation.
        hop_size (int): Hop size between frames.
        eps (float): Small value to avoid division by zero.
    
    Returns:
        torch.Tensor: RMS envelope, shape [B, num_frames].
    """
    # Ensure audio is 2D (Batch x Time)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)  # Add batch dimension

    # Number of frames
    batch_size, num_samples = audio.shape
    num_frames = (num_samples - frame_size) // hop_size + 1

    # Create a window function
    window = torch.hann_window(frame_size).to(audio.device)  # Smooth window
    
    # Frame the signal and apply the window
    frames = F.unfold(
        audio.unsqueeze(1),  # Add channel dimension for unfold
        kernel_size=(1, frame_size),
        stride=(1, hop_size)
    ).squeeze(1)  # Remove channel dimension
    
    frames = frames.reshape(batch_size, frame_size, num_frames)  # Reshape to [B, frame_size, num_frames]
    windowed_frames = frames * window[:, None]  # Apply window

    # Compute RMS for each frame
    rms = torch.sqrt(torch.mean(windowed_frames**2, dim=1) + eps)  # Shape: [B, num_frames]
    return rms


if __name__ == "__main__":
    # Load an audio file
    audio, sr = torchaudio.load("audios/out/out.wav")
    # Extract RMS envelope
    rms = rms_envelope(audio)
    plt.figure()
    plt.plot(rms.numpy()[0,:])
    plt.show()
    plt.savefig('rms.png')
    
    plt.figure()
    plt.plot(audio.numpy()[0,:])
    plt.show()
    plt.savefig('audio.png')

