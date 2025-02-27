import torch
import torchaudio
import numpy as np
from scipy.signal import get_window
from scipy.signal import butter
import yaml
import argparse
import time
import csv

from mambaTF.models import JustMamba2  # Replace with your specific model class
from mambaTF.models import MambaCoder
from mambaTF.models import MambaNet

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.size(0) > 1:  # Convert to mono if needed
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    # Resample to 48000 Hz if necessary
    if sample_rate != 48000:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=48000
        )
        sample_rate = 48000
    waveform = waveform.to("cuda")
    return waveform, sample_rate

def divide_into_buffers(waveform, buffer_size, hop_size):
    num_buffers = (waveform.size(1) - buffer_size) // hop_size + 1
    buffers = [waveform[:, i*hop_size:i*hop_size+buffer_size] for i in range(num_buffers)]
    return buffers

def compute_model_on_buffer(buffer, model):
    buffer_tensor = torch.tensor(buffer, dtype=torch.float32)
    
    # Synchronize CUDA (if available) to get an accurate time measurement
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        output = model(buffer_tensor)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    # Log the inference time in a CSV file (each inference time on a new line)
    with open("inference_times.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([elapsed_time])
    
    return output.cpu().numpy()

def overlap_add(buffers, hop_size):
    buffer_size = buffers[0].shape[1]
    output_length = (len(buffers) - 1) * hop_size + buffer_size
    output = np.zeros(output_length)
    window = get_window('hann', buffer_size)
    
    for i, buffer in enumerate(buffers):
        start = i * hop_size
        end = start + buffer_size
        output[start:end] += buffer.squeeze() * window
    
    return output

def psola_algorithm(waveform, sample_rate, model, buffer_size=1024, hop_size=512):
    buffers = divide_into_buffers(waveform, buffer_size, hop_size)
    processed_buffers = [compute_model_on_buffer(buffer, model) for buffer in buffers]
    output_waveform = overlap_add(processed_buffers, hop_size)
    return output_waveform

def normalize_tensor_wav(wav_tensor, low_threshold):
    max_val = torch.max(torch.abs(wav_tensor))
    if max_val > low_threshold:
        return wav_tensor / max_val
    else:
        return wav_tensor

def apply_highpass_filter(waveform, sample_rate, cutoff_freq=100.0):
    b, a = butter(
        N=4,  # 4 poles
        Wn=cutoff_freq,  # Cutoff frequency
        btype="highpass",  # High-pass filter
        fs=sample_rate,  # Sample rate
    )
    # Convert filter coefficients to PyTorch tensors with dtype=torch.float64
    b_tensor = torch.tensor(b, dtype=torch.float64).to(waveform.device)
    a_tensor = torch.tensor(a, dtype=torch.float64).to(waveform.device)
    
    # Convert waveform to torch.float64
    waveform = waveform.to(dtype=torch.float64)
    
    return torchaudio.functional.lfilter(waveform, a_tensor, b_tensor)

def main(file_path, out_path, checkpoint_path, config_path):
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    model = JustMamba2(**config["audionet"]["audionet_config"])
    checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    adjusted_state_dict = {
        k.replace("audio_model.", ""): v for k, v in checkpoint["state_dict"].items()
    }

    model.load_state_dict(adjusted_state_dict)
    if torch.cuda.is_available():
        model = model.cuda()

    waveform, sample_rate = load_audio(file_path)
    waveform = normalize_tensor_wav(waveform, 0.02)
    waveform = 0.7 * waveform
    buffer_size = int(config["constants"]["sample_rate"] * config["constants"]["slice"])
    output_waveform = psola_algorithm(waveform, sample_rate, model, buffer_size=buffer_size, hop_size=buffer_size // 2)

    # Apply high-pass filter at 100 Hz
    output_waveform = apply_highpass_filter(torch.tensor(output_waveform, dtype=torch.float64).unsqueeze(0), sample_rate, cutoff_freq=100.0)

    # Save the filtered output
    torchaudio.save(out_path, output_waveform, sample_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio using a Mamba model.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input audio file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output audio file.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the model configuration file.")

    args = parser.parse_args()

    main(args.input_path, args.output_path, args.checkpoint_path, args.config_path)
