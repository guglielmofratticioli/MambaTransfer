#!/bin/bash

# Check if config directory is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <config_directory>"
    exit 1
fi

config_dir="$1"

# Validate directory exists
if [ ! -d "$config_dir" ]; then
    echo "Error: Directory '$config_dir' does not exist"
    exit 1
fi

declare -a processed_files

while true; do
    new_files_found=0
    
    # Process YAML files in alphabetical order
    for file in "$config_dir"/*.yml; do
        # Skip non-existent files (case where no yaml files exist)
        [ -e "$file" ] || continue
        
        # Check if file has already been processed
        if [[ ! " ${processed_files[@]} " =~ " $file " ]]; then
            echo "Starting training with: $file"
            python train.py --conf_dir "$file"
            
            # Add to processed files
            processed_files+=("$file")
            new_files_found=1
        fi
    done

    # Exit loop if no new files were processed
    if [ $new_files_found -eq 0 ]; then
        echo "No new configuration files found. Exiting."
        break
    fi
done