import torch
import torchaudio
import numpy as np
from scipy.signal import get_window
import yaml
import argparse

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
    with torch.no_grad():
        output = model(buffer_tensor)
    return output.cpu().numpy()

def overlap_add(buffers, hop_size, window_type='hann'):
    buffer_size = buffers[0].shape[1]
    output_length = (len(buffers) - 1) * hop_size + buffer_size
    output = np.zeros(output_length)
    window = get_window(window_type, buffer_size)
    
    for i, buffer in enumerate(buffers):
        start = i * hop_size
        end = start + buffer_size
        output[start:end] += buffer.squeeze() * window
    
    return output

def psola_algorithm(waveform, sample_rate, model, buffer_size=1024, hop_size=512, window_type='hann'):
    buffers = divide_into_buffers(waveform, buffer_size, hop_size)
    processed_buffers = [compute_model_on_buffer(buffer, model) for buffer in buffers]
    output_waveform = overlap_add(processed_buffers, hop_size, window_type)
    return output_waveform

def normalize_tensor_wav(wav_tensor, low_threshold):
    max_val = torch.max(torch.abs(wav_tensor))
    if max_val > low_threshold:
        return wav_tensor / max_val
    else:
        return wav_tensor

def apply_triangular_windowing(output1, output2, hop_samples):
    # Create triangular window
    window = get_window('triang', hop_samples)
    hop_size = hop_samples // 2  # 50% overlap
    
    def process_signal(signal):
        frames = []
        for i in range(0, len(signal), hop_size):
            start = i
            end = start + hop_samples
            frame = signal[start:end]
            if len(frame) < hop_samples:
                frame = np.pad(frame, (0, hop_samples - len(frame)), mode='constant')
            frames.append(frame * window)
        # Overlap-add
        processed = np.zeros(len(signal) + hop_samples)
        for i, frame in enumerate(frames):
            start = i * hop_size
            end = start + hop_samples
            processed[start:end] += frame
        return processed[:len(signal)]
    
    windowed1 = process_signal(output1)
    windowed2 = process_signal(output2)
    return windowed1 + windowed2

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
    
    # Process original audio
    output1 = psola_algorithm(waveform, sample_rate, model, buffer_size=buffer_size, hop_size=buffer_size // 2)
    
    # Create padded version
    hop_seconds = config['audionet']['audionet_config']['hop']
    pad_length = int(hop_seconds  * 0.5 * sample_rate)
    padded_waveform = torch.nn.functional.pad(waveform, (pad_length, 0))  # Pad at the beginning
    
    # Process padded audio
    output2_padded = psola_algorithm(padded_waveform, sample_rate, model, buffer_size=buffer_size, hop_size=buffer_size // 2)
    output2 = output2_padded[pad_length:]  # Trim padding to align with original
    
    # Ensure equal length
    min_length = min(len(output1), len(output2))
    output1 = output1[:min_length]
    output2 = 0*output2[:min_length]
    
    # Apply triangular windowing and sum
    hop_samples = pad_length  # Derived from config hop and sample rate
    final_output = apply_triangular_windowing(output1, output2, hop_samples)
    
    # Save result
    torchaudio.save(out_path, torch.tensor(final_output).unsqueeze(0), sample_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio using a Mamba model.")
    parser.add_argument("--input_path", type=str, required=False, help="Path to the input audio file.", default="Inputs/Flute Solo.wav")
    parser.add_argument("--output_path", type=str, required=False, help="Path to save the output audio file.", default="Outputs/NoTremolo2.wav")
    parser.add_argument("--checkpoint_path", type=str, required=False, help="Path to the model checkpoint file.", default="FINAL_MODS/HOP_1/F2str/epoch=3377-step=280374.ckpt")
    parser.add_argument("--config_path", type=str, required=False, help="Path to the model configuration file.", default="FINAL_MODS/HOP_1/F2str/config.yml")

    args = parser.parse_args()

    main(args.input_path, args.output_path, args.checkpoint_path, args.config_path)