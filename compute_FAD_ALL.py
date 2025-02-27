import os
from frechet_audio_distance import FrechetAudioDistance

MODELS_DIR = "TestALL/HOP_1"
TEST_DIR = "TEST_FILES"
RESULTS_FILE = "FAD_HOP_01-LC.csv"

# Model configuration (modify these according to your needs)
# FAD_CONFIG = {
#     "model_name": "vggish",       # Options: "vggish", "pann", "clap", "encodec"
#     "sample_rate": 16000,         # 16000 for vggish/pann, 48000 for clap/encodec
#     "use_pca": False,
#     "use_activation": False,
#     "verbose": True
# }

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
    frechet = FrechetAudioDistance(
    use_pca=False,
    use_activation=False,
    verbose=False
)

    
    # Create results file
    with open(RESULTS_FILE, "w") as f:
        pass
    

    fad_score = frechet.score(
        MODELS_DIR, 
        TEST_DIR,
        dtype="float32"
    )
    
    # Save results
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{fad_score:.4f}\n")
    
        print(f"Processed - FAD: {fad_score:.4f}")
if __name__ == "__main__":
    main()