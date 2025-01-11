#100 copies of each chromosome of the combined negative sample as 1 copy
import numpy as np
'''
  "folder_path" is the output (-o) path directory of get_trainnegative_sample.py, which is the directory of the .npy file of the generated negative sample;
  "output_file_path" is the output path of merge_neggative, 
  "array1" is the output path of the merge_neggative. it is the negative.npy file path ;
  "array2" is the positive.npy file path.
  "output" is merge_positive_negative.py's output,it merged the positive.npy and negative.npy as pos_neg.npy
'''
def merge_neggative():
    a = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']
    res=5
    matrix_size=21
    for n in range(len(a)):
        folder_path = '/path_of_negative_sample0-100_dir/'
        output_file_path = f'path/of/negative_sample_dir/KR_{res}kb_matrix_chr' + str(a[n]) + '_negative.npy'
        merged_data = None

        for i in range(100):  
            file_name = f'KR_{res}kb_matrix_chr' + str(a[n]) + '_negative_'+ str(i) + '.npy'
        
            file_path = folder_path + file_name    
            try:
            
                data = np.load(file_path)        
        
                if merged_data is None:
                    merged_data = data
                else:
                
                    merged_data = np.vstack((merged_data, data))
            except FileNotFoundError:
            
                print(f"文件 {file_name} 不存在，跳过。")

        np.save(output_file_path, merged_data)
# #删选infy<=1
# def remove_infy1() :
#     data = 'path/of/negative_sample.npy'#merge_neggative()
#     outfile = 'path/of/negative_sample_delete.npy'
#     column_223 = data[:, 222]
#     rows_to_delete = np.where(column_223 <= 1)
#     filtered_data = np.delete(data, rows_to_delete, axis=0)
#     np.save(outfile, filtered_data) 

def merge_posneg():
    a = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']
    res=5
    matrix_size=21
    for n in range(len(a)):
        array1 = np.load(f'/path_of_negativenpy_sample_delete_dir/KR_{res}kb_matrix_chr' + str(a[n]) + '_negative.npy')# merge_neggative()'s output file
        array2 = np.load('/path_of_positivenpy_sample_dir/KR_{res}kb_matrix_chr' + str(a[n]) + '_positive.npy'')
        combined_array = np.vstack((array1, array2))
        print("merge data shape：", combined_array.shape)
        #np.random.shuffle(combined_array)
        output = '/path_of_merge_posneg_.npy/KR_{res}kb_matrix_chr' + str(a[n]) + '_pos_neg_tive.npy'
        np.save(output, combined_array)

def main():
    merge_neggative()
    #remove_infy1()
    merge_posneg()
