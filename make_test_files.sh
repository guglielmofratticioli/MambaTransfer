#!/bin/bash

MODELS_DIR="FINAL_MODS/HOP_05_Baby"
TEST_DIR="/nas/home/gfraticcioli/datasets/starnet/test"

# Process each model subfolder
find "$MODELS_DIR" -maxdepth 1 -type d | while read -r model_dir; do
    # Skip the parent MODELS directory
    [[ "$model_dir" == "$MODELS_DIR" ]] && continue
    
    model_name=$(basename "$model_dir")
    output_dir="$model_dir/output"
    
    # Determine input test directory based on model name
    case $model_name in
        F2str|F2str_Baby) input_test_subdir="Flutes" ;;
        str2F|str2F_Baby) input_test_subdir="Strings" ;;
        P2Xyl|P2Xyl_Baby) input_test_subdir="Piano" ;;
        Xyl2P|Xyl2P_Baby) input_test_subdir="Xylophone" ;;
        *) echo "Unknown model: $model_name"; continue ;;
    esac
    
    input_test_dir="$TEST_DIR/$input_test_subdir"
    
    # Verify required files and directories exist
    config_path="$model_dir/config.yml"
    ckpt_path=$(ls "$model_dir"/*.ckpt 2>/dev/null)
    if [ ! -f "$config_path" ] || [ -z "$ckpt_path" ]; then
        echo "Missing config or checkpoint in $model_name"
        continue
    fi
    
    if [ ! -d "$input_test_dir" ]; then
        echo "Input directory missing: $input_test_dir"
        continue
    fi
    
    # Process each audio file in the test directory
    find "$input_test_dir" -type f -name "*.wav" | while read -r audio_file; do
        output_file="$output_dir/$(basename "$audio_file")"
        
        echo "Processing $model_name with $(basename "$audio_file")"
        
        python Timbre_Transfer.py \
            --input_path "$audio_file" \
            --output_path "$output_file" \
            --checkpoint_path "$ckpt_path" \
            --config_path "$config_path"
    done
done