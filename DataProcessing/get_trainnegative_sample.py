import numpy as np
import datetime
import time
def count_rows(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        return len(lines)
import numpy as np
import datetime
import time
def get_submatrix_negative(matrix_input_file_path, center_point_input_file_path, output_file_path, negative_name_sort,chromosome,matrix_size):
    now = datetime.datetime.now()
    print("染色体", center_point_input_file_path.split('/')[-1], "负样本已开始", "    Current time is:", now)
    all_matrix_file = open(matrix_input_file_path, 'r')
    number_all_matrix_file = 0
    for line in all_matrix_file:
        number_all_matrix_file += 1
    all_matrix_file.close()

    center_point_file = open(center_point_input_file_path, 'r')
    point_list = []

    #center_point_interaction_frequency_path = {}
    # infile = open(interaction_frequency_path, 'r')
    # for line in infile:
    #     line = line.strip('\n').split('\t')
    #     center_point_interaction_frequency_path[str(int(int(line[0])/(res*1000) + 1)) + ',' + str(int(int(line[1]) / (res*1000) + 1))] = float(line[2])
    # infile.close()

    num_center_point_all = 0
    num_center_point_delete = 0
    for line_center_point_file in center_point_file:
        num_center_point_all += 1
        temp = line_center_point_file.strip('\n').split('	')

        temp_center_point = temp[0] + ',' + temp[1]

        #if temp_center_point in center_point_interaction_frequency_path and center_point_interaction_frequency_path[temp_center_point] > 1:
        point_list.append([int(temp[0]), int(temp[1])])
        num_center_point_delete += 1

    print("删除前中心点共有", num_center_point_all, "个，删除后有", num_center_point_delete, "个")
    point_list_temp = sorted(point_list)

    num_parts = 100  # 拆分成的份数
    part_length = len(point_list_temp) // num_parts
    point_list_all = [[] for _ in range(num_parts)]

    for i in range(num_parts):
        start_index = i * part_length
        end_index = (i + 1) * part_length
        point_list_all[i] = point_list_temp[start_index:end_index]

    if len(point_list_temp) % num_parts != 0:
        point_list_all[-1].extend(point_list_temp[num_parts * part_length:])

    new_number_point_all = 0
    for point_list_part in range(100):
        point_list = point_list_all[point_list_part]
        number_point = len(point_list)

        all_matrix = []
        for i in range(number_point):

            submatrix = [[0.0] * matrix_size for _ in range(matrix_size)]
            all_matrix.append(submatrix)
        # all_matrix = np.array(all_matrix)
        # print(all_matrix.shape)

        current_row = 0
        all_matrix_file = open(matrix_input_file_path, 'r')
        for line_all_matrix_file in all_matrix_file:
            current_row += 1
            if current_row % 100 == 0:
                print("负样本 ：", center_point_input_file_path.split('/')[-1], "第", point_list_part, "部分已经进行了", (current_row * 100) / number_all_matrix_file, "%")
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
                        current_column = 1  # 列从1开始
                    if end_column > len(line):
                        end_column = len(line)
                    while current_column <= end_column:
                    
                        all_matrix[num][current_row - start_row][current_column - start_column] = line[current_column - 1]
                     
                        current_column += 1
               
                elif current_row < start_row:
                    break

        all_matrix_file.close()
        center_point_file.close()
        delete_num = 0
        point_list_new = []
        all_matrix_new = []

        for num in range(len(all_matrix)):
            flag = 0
            for i in range(len(all_matrix[num])):
                if flag == 1:
                    break
                for j in range(len(all_matrix[num][i])):
                    if all_matrix[num][i][j] != '0.0' and all_matrix[num][i][j] != 0.0:  
                        flag = 1
                        break
            if flag == 1:  
                point_list_new.append(point_list[num])
                all_matrix_new.append(all_matrix[num])
            else:
                delete_num += 1
        print("原本有矩阵：", len(all_matrix))
        del all_matrix
        del point_list
        new_number_point = len(point_list_new)
        new_number_point_all += new_number_point
        print("共删除了：", delete_num, "个")
        print("还剩下负样本交互矩阵：", len(all_matrix_new), "个")
        print("还剩下负样本交互中心点：", len(point_list_new), "个")
        # point_list = [value for index, value in enumerate(point_list) if index not in delete_num]
        # all_matrix = [value for index, value in enumerate(all_matrix) if index not in delete_num]
        output_file_point_list = open(negative_name_sort[:-4] + '_' + str(point_list_part) + '.txt' , 'w+')
        for num in point_list_new:
            output_file_point_list.write('chr' + chromosome + '\t' + str(num[0]) + '\t' + str(num[1]) + '\n')
        output_file_point_list.close()

        # len_all_matrix = len(all_matrix)
        # for i in range(100)
        # all_matrix_temp = all_matrix[:len_all_matrix/100]
        all_matrix_new = np.array(all_matrix_new)
        all_matrix_new = all_matrix_new.astype('float32')
        all_matrix_new = all_matrix_new.reshape(-1, matrix_size*matrix_size)
        label_1 = np.zeros((new_number_point, 1))
        label_1 = label_1.astype('float32')
        all_all = np.concatenate((all_matrix_new, label_1), axis=1)
        np.save(output_file_path[:-4] + "_" + str(point_list_part) + '.npy', all_all)

    now = datetime.datetime.now()
    print("染色体", center_point_input_file_path.split('/')[-1], "负样本已结束", " Current time is:", now)   
   

a = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']
res=5
matrix_size=21
for n in range(len(a)):
    point_name = 'path/of/negative_centerpoint.txt'
    big_matrix_name = 'path/of/hicbigmatrix.chr'
    negative_center_delete_sort = 'path/of/save/negative_centerpoint-sort.txt'
    np_save_name = 'path/of/negative_sample.npy'
    np_save_name = 'path/of/negative_sample_delete.npy'
    get_submatrix_all_neg(big_matrix_name, point_name, np_save_name,negative_center_delete_sort,str(a[n]),matrix_size)
    