# Introduction
CGLoop: A Neural Network Framework for Chromatin Loop Prediction.
# Installation
CGLoop requires Python3 and several scientific packages to run.  
conda create -n CGLoop python=3.7.12   
conda activate CGLoop    
git clone https://github.com/wangyang199897/CD-Loop.git  
To set up a conda virtual environment, the following installation packages are required:  
scikit-learn=1.0.2   
tensorflow=2.11.0  
hic-straw=1.3.1  
numpy  
scipy   
pandas   
h5py   
cooler  
juicer_tools
# Usage
## Data preparation
Data preparation mainly involves: downloading .hic file, extracting Hi-C contact matrix from.hic file, and generating submatrix from Hi-C contact matrix. HTC data downloaded from [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525]()  
### Extracting Hi-C contact matrix from.hic file
Change the.jar,.hic, and output file paths in the GetBigMatrix_Cells_KRobserved.sh file, and run:  
```
bash GetBigMatrix_Cells_KRobserved.sh
```
The above process obtains the hic contact matrix for each chromosome from the.hic data.  
Modify the path to the input and output files in the Getnpymatrix_chr_all_sample.sh file, where the input file is the output file from the previous step, and run:  
### Generating submatrix from Hi-C contact matrix
```
bash Getnpymatrix_chr_all_sample.sh  
```
The above process cuts the hic contact matrix of each chromosome into multiple submatrices.
