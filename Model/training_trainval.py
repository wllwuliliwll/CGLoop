import os
import numpy as np
input_folder = "/path/of/posneg/dir/"
output_folder = "/path/of/outputfile/dir/"
#文件名匹配
file_template = "/name/of/posnegfile/chr{}_posneg.npy"
start_chr = 1
end_chr = 19

concatenated_data = []

for i in range(start_chr, end_chr+1):
    file_name = file_template.format(i)
    file_path = os.path.join(input_folder, file_name)
    data = np.load(file_path)
    concatenated_data.extend(data)
output_file_name = "chr1-19posneg.npy"
output_file_path = os.path.join(output_folder, output_file_name)
np.save(output_file_path, np.array(concatenated_data))

#将合并抽取的数据划分训练集和验证集
import numpy as np
from sklearn.model_selection import train_test_split
file_path = "/path/of/chr1-19posneg.npy"
array = np.load(file_path)
num_rows, num_cols = array.shape
labels = array[:, -1]
array1 = array[labels == 1]
array2 = array[labels == 0]

train_array1, val_array1 = train_test_split(array1, test_size=0.2, random_state=42)
train_array2, val_array2 = train_test_split(array2, test_size=0.2, random_state=42)

train = np.concatenate([train_array1, train_array2])
val = np.concatenate([val_array1, val_array2])

train_label_counts = np.sum(train[:, -1] == 1), np.sum(train[:, -1] == 0)
# print("Train Label Counts (1, 0):", train_label_counts)
val_label_counts = np.sum(val[:, -1] == 1), np.sum(val[:, -1] == 0)
# print("Validation Label Counts (1, 0):", val_label_counts)
# print("Shape of train:", train.shape)
# print("Shape of val:", val.shape)
np.save("/path/of/train.npy",train)
np.save("/path/of/val.npy", val)

