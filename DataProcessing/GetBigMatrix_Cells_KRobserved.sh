#!/bin/bash
for i in {20..22}
do
java -jar "/path/of/juicer_tools.jar" dump observed KR "path/of/file.hic" $i $i BP 5000 /path/of/frequence_matrix/KR_matrix_5kb.chr${i}_tmp -d
python remove_nan.py /path/of/frequence_matrix/KR_matrix_5kb.chr${i}_tmp /path/of/frequence_matrix/KR_matrix_5kb.chr$i
echo "chr"$i"已完成"
done
