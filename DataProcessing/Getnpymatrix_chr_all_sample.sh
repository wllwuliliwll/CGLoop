#!/bin/bash
checkMakeDirectory(){
    echo -e "checking directory: $1"
    if [ ! -e "$1" ]; then
        echo -e "\tmakedir $1"
        mkdir -p "$1"
    fi
}

chromList="17 18 19 1 6 12 2 3 4 5 7 8 9 10 11  13 14 15 16 17 18 19¡°
resolutions="5000"
DPATH="/public_data/wulili/data/HiC/Rwa_Cell/"
#CELL="GM12878 IMR90 K562 NHEK HMEC HUVEC"
CELL="mESC.NEW1"
for cell in $CELL; do
    echo $cell
    cell_Dir="${DPATH}${cell}"
    #mkdir -p $
    for resolution in $resolutions; do
        echo $resolution
        display_reso=$((resolution / 1000))
        mkdir -p "${cell_Dir}/${display_reso}kb/chr_all_samples"
        for chrom in $chromList; do
            echo $chrom
            python chr_all_sample.py ${cell_Dir}/${display_reso}kb/${display_reso}k_KR.chr$chrom ${cell_Dir}/${display_reso}kb/chr_all_samples/chr${chrom}_matrixsize${matrix_size}.npy 21 ${display_reso}
            python control_contact.py ${cell_Dir}/${display_reso}kb/chr_all_samples/chr${chrom}_matrixsize${matrix_size}.npy ${cell_Dir}/${display_reso}kb/chr_all_samples/chr${chrom}_matrixsize${matrix_size}_delet1.npy
        done
    done
done