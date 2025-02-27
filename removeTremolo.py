import numpy as np
from scipy.signal import butter, filtfilt
import librosa
import matplotlib.pyplot as plt

# Load the WAV file
audio, fs = librosa.load("audios/out/out.wav", sr=None)  # sr=None keeps the original sampling rate

# FFT to analyze frequency content
fft_signal = np.fft.fft(audio)
frequencies = np.fft.fftfreq(len(audio), d=1/fs)

# Set frequencies below 10 Hz to zero
fft_signal[np.abs(frequencies) < 10] = 1

# Plot the FFT of the modified audio
plt.figure(figsize=(12, 6))
plt.plot(frequencies, np.abs(fft_signal))
plt.title("FFT of the Audio Signal with Frequencies Below 10 Hz Set to Zero")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, fs/1000)  # Plot only the positive frequencies
plt.show()
plt.savefig('fft.png')
# Inverse FFT to get the modified audio signal
modified_audio = np.fft.ifft(fft_signal)

# Plot the waveform of the modified audio
plt.figure(figsize=(12, 6))
plt.plot(np.real(modified_audio))
plt.title("Waveform of the Modified Audio Signal")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.show()
plt.savefig('waveform.png')