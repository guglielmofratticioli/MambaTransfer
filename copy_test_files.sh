#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_directory> <destination_directory>"
    exit 1
fi

# Assign arguments to variables
SOURCE_DIR="$1"
DEST_DIR="$2"

# Check if the source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Source directory does not exist: $SOURCE_DIR"
    exit 1
fi

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Find all .wav files in the source directory and its subdirectories and copy them to the destination directory
find "$SOURCE_DIR" -type f -name "*.wav" -exec cp {} "$DEST_DIR" \;

echo "All .wav files have been copied to $DEST_DIR"