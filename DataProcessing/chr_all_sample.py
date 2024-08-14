import numpy as np
import datetime
import time
# def count_rows(filename):
#     with open(filename,  'r') as f:
#         count = 0
#         for line in f:
#             count += 1
#         return count
def count_rows(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        return len(lines)
#生成整条染色体所有样本
def get_submatrix_all(bigmatrix_file,save_file,matrix_size,res):
    matrix_size = int(matrix_size)
    res = int(res)
    now = datetime.datetime.now()
    lower = 30000
    highter = 3000000
    all = []
    lenth = count_rows(bigmatrix_file)
    print(lenth)
    for i in range(1,lenth+1):
        end1 = int(i + (lower / (res*1000)))
        end2 = int(i + (highter / (res*1000)))
        if end1 <= lenth and end2 <= lenth:
            for j in range(end1,end2+1):
                all.append([i,j])
        elif end1 <= lenth and end2 > lenth:
            for j in range(end1,lenth+1):
                all.append([i,j])
        else:
            break
    all = np.array(all)
    #print(all.shape)
    print("num of this chr centerpoints：",len(all))
    print("样本已开始", "Current time is:", now)
    num_center_point_all = 0
    num_center_point_delete = 0
    point_list = []
    for line_center_point_file in all:
        #print(line_center_point_file)
        point_list.append([int(line_center_point_file[0]), int(line_center_point_file[1])])
    #print(len(point_list))
    point_list_temp = sorted(point_list)
#拆分成100份，每份part_length个
    num_parts = 100  # 拆分成的份数
    part_length = len(point_list_temp) // num_parts
    point_list_all = [[] for _ in range(num_parts)]   
    for i in range(num_parts):
        start_index = i * part_length
        end_index = (i + 1) * part_length
        point_list_all[i] = point_list_temp[start_index:end_index]

    if len(point_list_temp) % num_parts != 0:
        point_list_all[-1].extend(point_list_temp[num_parts * part_length:])
    point_list = []
    new_number_point_all = 0
    merged_data = None
    for point_list_part in range(100):
        point_list = point_list_all[point_list_part]
        number_point = len(point_list)
        all_matrix = []
        for i in range(number_point):
            submatrix = [[0.0] * matrix_size for _ in range(matrix_size)]
            all_matrix.append(submatrix)
        current_row = 0
        all_matrix_file = open(bigmatrix_file, 'r')
        for line_all_matrix_file in all_matrix_file:
            current_row += 1
            if current_row % 100 == 0:
                print("第", point_list_part, "部分已经进行了", (current_row * 100) / lenth, "%")
            line = line_all_matrix_file.split('\t')
            for num in range(number_point):
                rows_point = point_list[num][0]
                columns_point = point_list[num][1]
                start_row = rows_point - (matrix_size//2)
                end_row = rows_point + (matrix_size//2)
                start_column = columns_point - (matrix_size//2)
                end_column = columns_point + (matrix_size//2)
                if start_row <= current_row <= end_row:
                    current_column = start_column
                    if start_column < 1:
                        current_column = 1  # Columns start from 1
                    if end_column > len(line):
                        end_column = len(line)
                    while current_column <= end_column:
                        all_matrix[num][current_row - start_row][current_column - start_column] = line[current_column - 1]

                        current_column += 1
                    
                elif current_row < start_row:
                    break
        all_matrix_file.close()
        #center_point_file.close()    
        
        all_matrix_new = np.array(all_matrix)
        all_matrix_new = all_matrix_new.astype('float32')
        all_matrix_new = all_matrix_new.reshape(-1, matrix_size*matrix_size)
        point_list_new = np.array(point_list)
        point_list_new = point_list_new.astype('float32')
        point_list_new = point_list_new.reshape(-1, 2)
        all_all = np.concatenate((point_list_new,all_matrix_new), axis=1)
        if merged_data is None:
            merged_data = all_all
        else:
            merged_data = np.vstack((merged_data, all_all))
    print(merged_data.shape)
    np.save(save_file, merged_data)
    now = datetime.datetime.now()
    print("样本已结束", " Current time is:", now)      
import sys, os
print(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
get_submatrix_all(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])

