#合并按比例的负样本每条染色体100份文件
import numpy as np
def merge_neggative():
    a = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']
    res=5
    matrix_size=21
    for n in range(len(a)):
        folder_path = 'path/of/negative_sample/dir/'
        output_file_path = 'path/of/negative_sample.npy'
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
#删选infy<=1
def remove_infy1() :
    data = 'path/of/negative_sample.npy'
    outfile = 'path/of/negative_sample_delete.npy'
    column_223 = data[:, 222]
    rows_to_delete = np.where(column_223 <= 1)
    filtered_data = np.delete(data, rows_to_delete, axis=0)
    np.save(outfile, filtered_data) 
#合并正负样本
def merge_posneg():
    a = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']
    res=5
    matrix_size=21
    for n in range(len(a)):
        array1 = np.load('path/of/egative_sample_delete.npy')
        array2 = np.load('path/of/positive_sample.npy')
        combined_array = np.vstack((array1, array2))
        print("合并后的数组形状：", combined_array.shape)
        np.random.shuffle(combined_array)
        np.save('path/of/merge_posneg_.npy', combined_array)

def main():
    merge_neggative()
    remove_infy1()
    merge_posneg()
