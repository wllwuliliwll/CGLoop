# Introduction
CGLoop: A Neural Network Framework for Chromatin Loop Prediction.
# Installation
CGLoop requires Python3 and several scientific packages to run.  
```
git clone https://github.com/wllwuliliwll/CGLoop.git 
conda create -n CGLoop python=3.7.12   
conda activate CGLoop    
```
To set up a conda virtual environment, the following installation packages are required:  
```
scikit-learn=1.0.2   
tensorflow=2.11.0  
hic-straw=1.3.1
joblib=1.3.1  
numpy  
scipy   
pandas   
h5py   
cooler
pysam  
juicer_tools
```
# Usage
## Data preparation
Data preparation mainly involves: downloading .hic file, extracting Hi-C contact matrix from.hic file, and generating submatrix from Hi-C contact matrix. Hi-C data downloaded from [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525]()  
### Extracting Hi-C contact matrix from.hic file
The process obtains the hic contact matrix for each chromosome from the.hic file. It will output the frequency_matrix file.  
Modify the path to the input and output files in the GetBigMatrix_Cells_KRobserved.sh file: The.jar file is the path where the juicer tools resides, and run:  
```
bash GetBigMatrix_Cells_KRobserved.sh
```
### Generating sub-matrix from Hi-C contact matrix
The process cuts the hic contact matrix of each chromosome into multiple submatrices.
Modify the path to the input and output files in the Getnpymatrix_chr_all_sample.sh file, where the input file is the output file from the previous step, DPATH is the root directory of the frequence_matrix file, and run:  
```
bash Getnpymatrix_chr_all_sample.sh  
```
## Model training
If you want to retrain the model, follow the training data generation method in our paper to get the required training sample. 
###  Get the norm_factor file, Interaction_frequency file, frequence_matrix file 
If you have not norm_factor file and Interaction_frequency file,please run the following command, the generated file will be used in the subsequent data generation process.
Here, you need to update the path in the corresponding file and run: 
```
GetKRnorm _excerces_factor.sh 
GetKRobserved_excerces.sh
```
###  Get the training positive samples 
Here, you need to download the gm12878.tang.ctcf-chiapet.hg19.bedpe and gm12878.mumbach.h3k27ac-hichip.hg19.bedpe files, the download address is in the supplemental file.And run:
```
python get_trainpositive_point.py  -i1 [ctcf.bedpe] -i2 [h3k27ac.bedpe] -o [PositiveTxt_dir] -p [processdata_dir]
python get_trainpositive_sample.py -d [PositiveTxt_dir] -b [frequence_matrix_dir] -o [Positivenpy_dir]  
```
###  Get the training negative samples  
```
python get_trainnegative_point.py -p [PositiveTxt_dir] -n [norm_factor_dir] -o [NegativeTxt_dir] 
python get_trainnegative_sample.py -x [frequence_matrix_dir] -p [NegativeTxt_dir] -o [Negativenpy_dir] -n [negative_name_sort_dir] -i [Interaction_frequency_dir] 
```
### Merge positive samples and negative samples  
Here, you need to update the path in the corresponding file and run:  
```
python merge_positive_negative.py  
```
### Get Train-validation-test sample  
Here, you need to update the path in the corresponding file and run: 
```
python training_trainval.py  
python training_test.py  
```
### Training  
```
python loops_train.py -t [Train file ] -v [validation file ] 
```
## Use CGLoop to predict chromatin loops  
Run the following code to make predictions of genome-wide chromatin loops:  
```
bash cgloop.sh
```
## Clustering
The clustering method in peakachu[https://github.com/tariks/peakachu]() was used for clustering screening:  
```
git clone https://github.com/tariks/peakachu
```
Update the .hic file paths, the .pkl file path, and the output file path, and the .pkl file download from [https://github.com/tariks/peakachu]().Then run:
```
bash peakachucluster.sh
```
# Output file format
```
[chrname_column1]            The chromosome name1 of the left anchor of the chromatin loop
[location10]                 The starting position of the left anchor of the chromatin loop
[location11]                 The end position of the left anchor of the chromatin loop
[chrname_column2]            The chromosome name2 of the right anchor of the chromatin loop
[location20]                 The starting position of the right anchor of the chromatin loop
[location21]                 The end position of the right anchor of the chromatin loop
[predictions]                The predictions of chromatin loop
[infy]                       The interaction strength of chromatin loop
```










