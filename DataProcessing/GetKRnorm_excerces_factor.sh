#!/bin/bash
for i in {20..22}
do
java -jar "/path/of/juicer_tools.jar" dump norm KR "/path/of/file.hic" $i BP 5000 /path/of/norm_factor/chr$i-5kb.KRnorm
echo "chr"$i"已完成"
done
