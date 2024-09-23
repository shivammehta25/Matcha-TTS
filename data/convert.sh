#!/bin/bash

new_sample_rate=44100

for file in before/*.wav; do
    output_file="wav/$(basename "$file")"
    
    ffmpeg -i "$file" -ar $new_sample_rate "$output_file"
    
    echo "Resampled $file to $output_file at $new_sample_rate Hz"
done
