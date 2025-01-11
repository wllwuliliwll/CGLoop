#!/bin/bash
for i in {20..22}
do
java -jar "/path/of/juicer_tools.jar" dump observed KR "/path/of/file.hic" $i $i BP 5000 /path/of/Interaction_frequency/chr$i-5kb.KRobserved
echo "chr"$i"已完成"
done
