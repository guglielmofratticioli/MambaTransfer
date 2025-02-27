#!/bin/bash

# Set the bitrate for MP3 conversion (adjust as needed)
BITRATE="192k"

# Find and convert all .wav files to .mp3
find . -type f -iname "*.wav" | while read -r file; do
    # Get the output file name by replacing .wav with .mp3
    output="${file%.wav}.mp3"

    # Convert the file
    ffmpeg -i "$file" -b:a $BITRATE "$output"

    # Check if conversion was successful
    if [ $? -eq 0 ]; then
        rm "$file"  # Delete the original .wav file
    else
        echo "Error converting $file"
    fi
done
