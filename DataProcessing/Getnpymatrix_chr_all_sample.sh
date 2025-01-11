#!/bin/bash
checkMakeDirectory(){
    echo -e "checking directory: $1"
    if [ ! -e "$1" ]; then
        echo -e "\tmakedir $1"
        mkdir -p "$1"
    fi
}

chromList="20 21 22 "
resolutions="5000"
DPATH="/path/of/frequence_matrix/"
save_path=$(dirname "$DPATH")/chr_all_sample
mkdir -p "$save_path"
matrix_size=21
for resolution in $resolutions; do
    echo $resolution
    display_reso=$((resolution / 1000))
    for chrom in $chromList; do
        echo $chrom
        python chr_all_sample.py ${DPATH}KR_matrix_${display_reso}kb.chr$chrom ${save_path}/chr${chrom}_matrixsize${matrix_size}_tmp.npy $matrix_size ${display_reso}
        python control_contact.py ${save_path}/chr${chrom}_matrixsize${matrix_size}_tmp.npy ${save_path}/chr${chrom}_matrixsize${matrix_size}.npy
    done
done
