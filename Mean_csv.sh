#!/bin/bash

# Check for filename argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 filename.csv"
    exit 1
fi

file="$1"

# Calculate the sum and count of numbers, then compute the average
awk '{
    sum += $1; 
    count++
} 
END { 
    if (count > 0) 
        printf "Average: %.4f\n", sum/count; 
    else 
        print "No data found." 
}' "$file"