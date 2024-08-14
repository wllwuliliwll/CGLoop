#!/bin/bash

# converting .hic files to .cool file
hic_files=(

    "/Path/to/predicte/chaname.hic"
)
resolutions="25000 50000"
for hic_file in "${hic_files[@]}"; do
    input_folder=$(dirname "$hic_file")
    filename=$(basename "$hic_file" .hic)
    echo "Converting $hic_file to .cool files..."
    for resolution in $resolutions; do

        output_file="$input_folder/${filename}.cool"
        hicConvertFormat -m "$hic_file" --inputFormat hic --outputFormat cool -o "$output_file" --resolutions "$resolution"
        
        echo "Generated $output_file"
    done
done

echo "Conversion completed."
