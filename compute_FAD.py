import os
from frechet_audio_distance import FrechetAudioDistance

MODELS_DIR = "FINAL_MODS/HOP_05_Baby"
TEST_DIR = "/nas/home/gfraticcioli/datasets/starnet/test"
RESULTS_FILE = "fad_results.csv"

# Model configuration (modify these according to your needs)
FAD_CONFIG = {
    "model_name": "vggish",       # Options: "vggish", "pann", "clap", "encodec"
    "sample_rate": 16000,         # 16000 for vggish/pann, 48000 for clap/encodec
    "use_pca": False,
    "use_activation": False,
    "verbose": True
}

# Mapping from model names to their reference directories
MODEL_REFERENCE_MAPPING = {
    "F2str": "Strings",
    "F2str_Baby": "Strings",
    "str2F": "Flutes",
    "str2F_Baby": "Flutes",
    "P2Xyl": "Xylophone",
    "P2Xyl_Baby": "Xylophone",
    "Xyl2P": "Piano",
    "Xyl2P_Baby": "Piano"
}

def main():
    # Initialize FAD calculator
    frechet = FrechetAudioDistance(**FAD_CONFIG)
    
    # Create results file
    with open(RESULTS_FILE, "w") as f:
        f.write("Model,ReferenceDir,NumFiles,FADScore\n")
    
    # Process each model
    for model_name in os.listdir(MODELS_DIR):
        model_path = os.path.join(MODELS_DIR, model_name)
        output_dir = os.path.join(model_path, "output")
        
        # Validate paths
        if not os.path.isdir(model_path):
            continue
        if not os.path.exists(output_dir):
            print(f"Skipping {model_name}: No output directory")
            continue
            
        # Get reference directory
        reference_subdir = MODEL_REFERENCE_MAPPING.get(model_name)
        if not reference_subdir:
            print(f"Skipping {model_name}: No reference mapping")
            continue
            
        reference_dir = os.path.join(TEST_DIR, reference_subdir)
        if not os.path.exists(reference_dir):
            print(f"Skipping {model_name}: Reference directory {reference_dir} not found")
            continue

        # Count audio files
        output_files = [f for f in os.listdir(output_dir) if f.endswith(".wav")]
        reference_files = [f for f in os.listdir(reference_dir) if f.endswith(".wav")]
        if not output_files or not reference_files:
            print(f"Skipping {model_name}: Missing audio files")
            continue

        # Compute FAD score
        try:
            fad_score = frechet.score(
                reference_dir, 
                output_dir,
                dtype="float32"
            )
            
            # Save results
            with open(RESULTS_FILE, "a") as f:
                f.write(f"{model_name},{reference_subdir},{len(output_files)},{fad_score:.4f}\n")
            
            print(f"Processed {model_name} - FAD: {fad_score:.4f}")
            
        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")

if __name__ == "__main__":
    main()