import os
import numpy as np
import librosa

class JaccardAudioDistance:
    def __init__(self, sr=16000, n_mels=128, threshold=-30, verbose=False):
        """
        sr: sample rate for loading audio
        n_mels: number of Mel bands for spectrogram
        threshold: dB threshold for binarizing the spectrogram
        verbose: whether to print progress details
        """
        self.sr = sr
        self.n_mels = n_mels
        self.threshold = threshold
        self.verbose = verbose

    def compute_binary_melspec(self, file_path):
        # Load the .wav file
        y, sr = librosa.load(file_path, sr=self.sr)
        # Compute the mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        # Convert power spectrogram (amplitude squared) to dB scale
        S_db = librosa.power_to_db(S, ref=np.max)
        # Binarize: True where energy exceeds the threshold
        binary_spec = S_db > self.threshold
        return binary_spec

    def score(self, models_dir, test_dir):
        # List files (assumes matching filenames/order)
        models_files = sorted(os.listdir(models_dir))
        test_files = sorted(os.listdir(test_dir))
        distances = []
        
        for m_file, t_file in zip(models_files, test_files):
            m_path = os.path.join(models_dir, m_file)
            t_path = os.path.join(test_dir, t_file)
            
            # Compute binary mel spectrograms for both files
            m_spec = self.compute_binary_melspec(m_path)
            t_spec = self.compute_binary_melspec(t_path)
            
            # Ensure the matrices have the same shape: crop to minimum number of frames
            min_frames = min(m_spec.shape[1], t_spec.shape[1])
            m_spec = m_spec[:, :min_frames]
            t_spec = t_spec[:, :min_frames]
            
            # Flatten the binary masks into 1D arrays
            m_flat = m_spec.flatten()
            t_flat = t_spec.flatten()
            
            # Compute the intersection and union of the binary vectors
            intersection = np.sum(np.logical_and(m_flat, t_flat))
            union = np.sum(np.logical_or(m_flat, t_flat))
            
            # Compute Jaccard distance; handle zero union case
            jd = 1 - (intersection / union) if union != 0 else 0.0
            distances.append(jd)
            
            if self.verbose:
                print(f"Processed {m_file} and {t_file} - Jaccard Audio Distance: {jd:.4f}")
        
        # Return the average Jaccard distance over all file pairs
        return np.mean(distances)

def main():
    # Directory paths (adjust to your folder structure)
    MODELS_DIR = "TestALL/HOP_05_Baby"
    TEST_DIR = "TEST_FILES"
    RESULTS_FILE = "JD_HOP_05_Baby.csv"
    
    # Configuration for the Jaccard Audio Distance evaluation
    config = {
        "sr": 16000,        # sample rate
        "n_mels": 128,      # number of mel bands
        "threshold": -30,   # dB threshold for binarization
        "verbose": True
    }
    
    # Initialize the JaccardAudioDistance calculator
    jad = JaccardAudioDistance(**config)
    
    # Create (or clear) the results file
    with open(RESULTS_FILE, "w") as f:
        pass
    
    # Compute the average Jaccard Audio Distance score
    score = jad.score(MODELS_DIR, TEST_DIR)
    
    # Save the score to the results file
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{score:.4f}\n")
    
    print(f"Processed - Jaccard Audio Distance: {score:.4f}")

if __name__ == "__main__":
    main()
