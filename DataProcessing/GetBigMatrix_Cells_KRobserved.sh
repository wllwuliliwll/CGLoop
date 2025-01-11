#!/bin/bash
for i in {20..22}
do
java -jar "/home/wulili/juicer_tools.jar" dump observed KR "/mnt/sda/wll/CGLoopData/HiC/Rwa_Cell/IMR90/GSE63525_IMR90_combined_30.hic" $i $i BP 5000 /mnt/sda/wll/CNNCNN-NewClear/IMR90Data/frequence_matrix/KR_matrix_5kb.chr${i}_tmp -d
python remove_nan.py /mnt/sda/wll/CNNCNN-NewClear/IMR90Data/frequence_matrix/KR_matrix_5kb.chr${i}_tmp /mnt/sda/wll/CNNCNN-NewClear/IMR90Data/frequence_matrix/KR_matrix_5kb.chr$i
echo "chr"$i"已完成"
done