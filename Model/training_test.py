#合并比例抽取的测试集
import os
import numpy as np
input_folder = "/path/of/posneg/dir/"
output_folder = "/path/of/test/dir/"
file_template = "/name/of/chr{}_posneg.npy"
start_chr = 20
end_chr = 22
concatenated_data = []

for i in range(start_chr, end_chr+1):

    file_name = file_template.format(i)
    file_path = os.path.join(input_folder, file_name)
    data = np.load(file_path)

    concatenated_data.extend(data)
output_file_name = "test.npy"
output_file_path = os.path.join(output_folder, output_file_name)
np.save(output_file_path, np.array(concatenated_data))