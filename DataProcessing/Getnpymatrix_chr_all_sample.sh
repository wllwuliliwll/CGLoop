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
cell_Dir="${DPATH}"
matrix_size=21
for resolution in $resolutions; do
    echo $resolution
    display_reso=$((resolution / 1000))
    mkdir -p "${cell_Dir}/${display_reso}kb"
    for chrom in $chromList; do
        echo $chrom
        python chr_all_sample.py ${cell_Dir}/KR_matrix_${display_reso}.chr$chrom ${cell_Dir}/${display_reso}kb/chr${chrom}_matrixsize${matrix_size}.npy $matrix_size ${display_reso}
        python control_contact.py ${cell_Dir}/${display_reso}kb/chr${chrom}_matrixsize${matrix_size}.npy ${cell_Dir}/${display_reso}kb/chr${chrom}_matrixsize${matrix_size}_delet1.npy
    done
done
