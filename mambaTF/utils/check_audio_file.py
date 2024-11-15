import argparse
import os
import matplotlib.pyplot as plt
import torchaudio

# Function to plot the first 1000 samples of the waveform
def plot_waveform(audio_path):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_path)

    # Limit the waveform to the first 1000 samples
    waveform = waveform[:,:]

    # Plot the waveform
    plt.figure(figsize=(10, 4))
    plt.plot(waveform.t().numpy())
    plt.title(f"First 1000 Samples of {os.path.basename(audio_path)}")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

# Main function to process the folder
def main(folder_path):
    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            audio_path = os.path.join(folder_path, filename)
            print(f"Processing file: {audio_path}")
            plot_waveform(audio_path)

# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot the first 1000 samples of waveforms in a folder.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing .wav files")
    args = parser.parse_args()

    # Call the main function with the folder path
    main(args.folder_path)
