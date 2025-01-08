#!/bin/bash
checkMakeDirectory(){
    echo -e "checking directory: $1"
    if [ ! -e "$1" ]; then
        echo -e "\tmakedir $1"
        mkdir -p "$1"
    fi
}
juicer_tool="/path/of/juicer_tools.jar"
#chromList="20 21 22"
chromList="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22"
#resolutions="5000 10000 25000 50000 100000"
resolutions="5000"
DPATH="/path/of/hicfile/dir/"
CELL="GM12878/GSE63525_GM12878_combined_republic_30.hic"
outputDir="/path/of/savedata/dir/"
for cell in $CELL; do
    cell_name=$(echo $cell | cut -d'/' -f1)
    echo $cell_name
    #hic_file=$(echo $cell | cut -d'/' -f2)
    cell_outputDir="${outputDir}/${cell_name}"
    #mkdir -p $
    for resolution in $resolutions; do
        display_reso=$((resolution / 1000))
        mkdir -p "${cell_outputDir}/${display_reso}kb"
        for chrom in $chromList; do
            java -jar $juicer_tool dump observed KR $DPATH$cell chr$chrom chr$chrom BP $resolution ${cell_outputDir}/${display_reso}kb/${display_reso}k_KR.chr${chrom}_tmp -d
            python remove_nan.py ${cell_outputDir}/${display_reso}kb/${display_reso}k_KR.chr${chrom}_tmp ${cell_outputDir}/${display_reso}kb/${display_reso}k_KR.chr${chrom}
        done
    done
done
