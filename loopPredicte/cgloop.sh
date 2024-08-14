#!/bin/bash

input_paths=(
"../Data/PredicteData/chr20.npy"
"../Data/PredicteData/chr21.npy"
"../Data/PredicteData/chr22.npy"

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
        #outputfolder_dir="../Data/PredicteData/"
        #mkdir -p $outputfolder_dir
        python ./loops_predicte.py -i $input_path -o ../Data/PredicteData/${chrname}.bedpe -r $resolution               
    done
done

