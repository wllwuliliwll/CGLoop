#!/bin/bash
for i in {20..22}
do
java -jar "/home/wulili/juicer_tools.jar" dump norm KR "/mnt/sda/wll/CGLoopData/HiC/Rwa_Cell/IMR90/GSE63525_IMR90_combined_30.hic" $i BP 5000 /mnt/sda/wll/CNNCNN-NewClear/IMR90Data/norm_factor/chr$i-5kb.KRnorm
echo "chr"$i"已完成"
done