import numpy as np
import scipy.signal as signal
import soundfile as sf
import matplotlib.pyplot as plt

def remove_tremolo(signal_in, fs=44100, mod_freq=20):  
    """
    Removes tremolo (amplitude modulation) from an audio signal using the Hilbert Transform.
    
    Parameters:
    - signal_in: np.array, input signal with tremolo
    - fs: int, sampling rate in Hz
    - mod_freq: float, estimated tremolo frequency in Hz
    
    Returns:
    - signal_out: np.array, signal with tremolo removed
    """
    eps = 1e-10
    # Apply Hilbert Transform to get analytic signal
    analytic_signal = signal.hilbert(signal_in)
    
    # Extract envelope (instantaneous amplitude)
    envelope = np.abs(analytic_signal)

    # Smooth the envelope using a low-pass filter
    b, a = signal.butter(4, mod_freq / (fs / 2), btype='low')
    smoothed_envelope = signal.filtfilt(b, a, envelope + eps)


    # Plot envelope and smoothed_envelope in different graphs of the same image
    #plt.figure()
    #plt.plot(envelope)
    #plt.plot(smoothed_envelope)
    #plt.savefig("smoothed_envelope.png")
    #plt.close()
    #
    #plt.figure()
    #plt.plot(normalized)
    #plt.savefig("envelope.png")
    #plt.close()

    # Normalize to remove tremolo
    signal_out = 0.1* signal_in / (smoothed_envelope + 1e-6)  # Avoid division by zero

    # RMS normalize signal_out based on signal_in's RMS
    rms_in = np.sqrt(np.mean(signal_in**2))  # RMS of input signal
    rms_out = np.sqrt(np.mean(signal_out**2))  # RMS of output signal
    #signal_out = signal_out * (rms_in / (rms_out + 1e-6))  # Scale to match RMS of signal_in

    return signal_out

def swap_envelope(signal_in,modulator, fs=44100, mod_freq=20):  
    """
    Removes tremolo (amplitude modulation) from an audio signal using the Hilbert Transform.

    Parameters:
    - signal_in: np.array, input signal with tremolo
    - fs: int, sampling rate in Hz
    - mod_freq: float, estimated tremolo frequency in Hz

    Returns:
    - signal_out: np.array, signal with tremolo removed
    """
    eps = 1e-10
    # Apply Hilbert Transform to get analytic signal
    analytic_signal = signal.hilbert(signal_in)
    analytic_signal_mod = signal.hilbert(modulator)
    
    # Extract envelope (instantaneous amplitude)
    envelope = np.abs(analytic_signal)
    envelope_mod = np.abs(analytic_signal_mod)

    # Smooth the envelope using a low-pass filter
    b, a = signal.butter(4, mod_freq / (fs / 2), btype='low')
    smoothed_envelope = signal.filtfilt(b, a, envelope +eps)
    smoothed_envelope_mod = signal.filtfilt(b, a, envelope_mod +eps)

    # plot envelope and smoothed_envelope in different graphs of same image



    # Normalize to remove tremolo
    #normalized = 0.1*signal_in / (smoothed_envelope + 1e-6)  # Avoid division by zero

    plt.figure()
    plt.plot(envelope)
    plt.plot(smoothed_envelope)
    plt.savefig("smoothed_envelope.png")
    plt.close()
    plt.figure()
    plt.plot(signal_in)
    plt.savefig("envelope.png")
    plt.close()
    signal_out = signal_in * smoothed_envelope_mod

    return signal_out


def fft_hard_cut(signal, sample_rate, cutoff=20000):
    N = len(signal)  # Number of samples
    freq = np.fft.fftfreq(N, d=1/sample_rate)  # Frequency bins
    spectrum = np.fft.fft(signal)  # FFT

    # Hard cut: Zero out frequencies above cutoff
    spectrum[np.abs(freq) > cutoff] = 0

    # Inverse FFT to get filtered signal
    filtered_signal = np.fft.ifft(spectrum).real  # Take the real part

    return filtered_signal



if __name__ == "__main__":
    # Load example audio files
    signal_1, fs = sf.read("/nas/home/gfraticcioli/projects/MambaTransfer/audios/out/3kEpo(S2F)_Sine_Output.wav")
    signal_2, _ = sf.read("/nas/home/gfraticcioli/projects/MambaTransfer/Inputs/SaxDDD.wav")
    
    signal_1 = fft_hard_cut(signal_1, fs)


    # Ensure both signals have the same length
    min_length = min(len(signal_1), len(signal_2))
    signal_1 = signal_1[:min_length]
    signal_2 = signal_2[:min_length]

    if signal_1.ndim > 1:  # Convert stereo to mono by averaging channels
        signal_1 = np.mean(signal_1, axis=1)

    if signal_2.ndim > 1:  # Convert stereo to mono by averaging channels
        signal_2 = np.mean(signal_2, axis=1)
    
    range = range(24*12288,26*12288)
    signal_1 = signal_1[range]
    signal_2 = signal_2[range]

    # Apply the modulation swap
    signal_1 = remove_tremolo(signal_1,fs,mod_freq=20)
    signal_1 = signal_1.clip(-1, 1)  # Clip to avoid clipping during playback
    out_signal = remove_tremolo(signal_1,fs,mod_freq=400)
    out_signal = out_signal.clip(-1, 1)  # Clip to avoid clipping during playback

    out_signal = swap_envelope(out_signal,signal_2,fs,mod_freq=20)
    # plot outsignal and save image
    plt.figure()
    plt.plot(out_signal)
    plt.savefig("out_signal.png")
    
    # Save the output
    sf.write("normalized_signal.wav", out_signal, fs)

