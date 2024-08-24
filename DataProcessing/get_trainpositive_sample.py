import numpy as np
import datetime
import time
def get_submatrix_positive(matrix_input_file_path, center_point_input_file_path, output_file_path,matrix_size):
    now = datetime.datetime.now()
    print("染色体", center_point_input_file_path.split('/')[-1], "正样本已开始", "    Current time is:", now)
    all_matrix_file = open(matrix_input_file_path, 'r')
    number_all_matrix_file = 0
    for line in all_matrix_file:
        number_all_matrix_file += 1
    all_matrix_file.close()

    center_point_file = open(center_point_input_file_path, 'r')
    point_list = []
    for line_center_point_file in center_point_file:
        temp = line_center_point_file.split('	')
        point_list.append([int(temp[1]), int(temp[2])])
    point_list = sorted(point_list)
    number_point = len(point_list)
    #print(number_point)
    all_matrix = []
    matrix_positions=[]
    for i in range(number_point):

        submatrix = [[0.0] * matrix_size for _ in range(matrix_size)]
        all_matrix.append(submatrix)
        matrix_positions.append((0, 0))
        
    current_row = 0
    all_matrix_file = open(matrix_input_file_path, 'r')
    for line_all_matrix_file in all_matrix_file:
        current_row += 1
        #print("current_row：",current_row)
        if current_row % 100 == 0:
            print("正样本：", center_point_input_file_path.split('/')[-1], "已经进行了",
                  (current_row * 100) / number_all_matrix_file, "%")
        line = line_all_matrix_file.split('\t')
      
        for num in range(number_point):#中心点文件行数
            #print("中心点：",num)
            rows_point = point_list[num][0]
            columns_point = point_list[num][1]
         
            start_row = rows_point - matrix_size//2
            end_row = rows_point + matrix_size//2
            start_column = columns_point - matrix_size//2
            end_column = columns_point + matrix_size//2
            if start_row <= current_row <= end_row:
                current_column = start_column
                if start_column < 1:
                    current_column = 1
                if end_column > len(line):
                    end_column = len(line)
                while current_column <= end_column:
            
                    all_matrix[num][current_row - start_row][current_column - start_column] = line[current_column - 1]
         
                    current_column += 1                    
            elif current_row < start_row:
                break

    all_matrix_file.close()
    center_point_file.close()
    
    all_matrix = np.array(all_matrix)
    all_matrix = all_matrix.astype('float32')
    all_matrix = all_matrix.reshape(-1, matrix_size*matrix_size)
    point_list = np.array(point_list)
    point_list = point_list.astype('float32')
    point_list = point_list.reshape(-1, 2)
    label_1 = np.ones((number_point, 1))
    label_1 = label_1.astype('float32')


    all_all = np.concatenate((all_matrix, label_1), axis=1)

    np.save(output_file_path, all_all)
    now = datetime.datetime.now()
    print("染色体", center_point_input_file_path.split('/')[-1], "正样本已结束", " Current time is:", now)
a = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
     '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']
res=5
matrix_size=21
for n in range(len(a)):
    positive_name = '/path/of/positive_centerpoint.txt'
    big_matrix_name = '/path/of/bigmatrix.chr'
    np_save_name_positive = '/path/of/positive_sample.npy'
    get_submatrix_positive(big_matrix_name,positive_name,np_save_name_positive,matrix_size)