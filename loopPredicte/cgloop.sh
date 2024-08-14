#!/bin/bash

input_paths=(
"/Path/to/chrname.npy"
)

resolutions=(5000)
for input_path in "${input_paths[@]}"
do
    for resolution in "${resolutions[@]}"
    do
        parts=(${input_path//\// })
        cell=${parts[5]}
        filename=${parts[-1]}
        chrname=$(echo $filename | grep -oP 'chr\d+')
        res=$((resolution / 1000))
        python ./loops_predicte.py -i $input_path -o /Path/to/${chrname}.bedpe -r $resolution               
    done
done

