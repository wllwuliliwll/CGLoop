import numpy as np
import datetime
import time
import argparse
import numpy as np
import os
def get_submatrix_positive(matrix_input_file_path, center_point_input_file_path, output_file_path,matrix_size):
    now = datetime.datetime.now()
    print( center_point_input_file_path.split('/')[-1], "start", "    Current time is:", now)
    all_matrix_file = open(matrix_input_file_path, 'r')
    number_all_matrix_file = 0
    for line in all_matrix_file:
        number_all_matrix_file += 1
    all_matrix_file.close()

    center_point_file = open(center_point_input_file_path, 'r')
    point_list = []
    for line_center_point_file in center_point_file:
        temp = line_center_point_file.split('	')
        point_list.append([int(temp[1]), int(temp[2])])#
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
        # if current_row % 100 == 0:
        #     print("正样本：", center_point_input_file_path.split('/')[-1], "已经进行了",
        #           (current_row * 100) / number_all_matrix_file, "%")
        line = line_all_matrix_file.split('\t')
        #line=np.array(line)
        #print(line.shape)
        for num in range(number_point):#中心点文件行数
            #print("center point：",num)
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
                    #print("current_column:",current_column)
                    #print(num)
                    all_matrix[num][current_row - start_row][current_column - start_column] = line[current_column - 1]
                    #all_matrix=np.array(all_matrix)
                    #print("all_matrix.shape:",all_matrix.shape)#(1110, 21, 21)
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
    all_all = np.concatenate((all_matrix, label_1), axis=1)#不带位置442列，训练集验证集，Positive时使用
    np.save(output_file_path, all_all)
    now = datetime.datetime.now()
    print(center_point_input_file_path.split('/')[-1], "end", " Current time is:", now)
def parse_args():
    default_chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                           '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']
    
    parser = argparse.ArgumentParser(description="Generate positive samples based on HiC matrix data.")
    parser.add_argument('-c', '--chromosomes', type=str, nargs='+', default=default_chromosomes, 
                        help="List of chromosomes (e.g., 1 2 3 ... X). Defaults to all chromosomes if not provided.")
    parser.add_argument('-r', '--resolution', type=int, default=5, help="Resolution for the matrix (e.g., 5, 10, etc.)")
    parser.add_argument('-m', '--matrix_size', type=int,default=21, help="Size of the matrix (e.g., 21)")
    parser.add_argument('-o', '--output_dir', type=str, required=True, help="Output directory to save results")
    parser.add_argument('-b', '--bigmatrix_dir', type=str, required=True, help="Input file of bigmatrix file")
    parser.add_argument('-d', '--data_dir', type=str, required=True, help="Directory where the big matrices and positive samples are located")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # 获取参数值
    chromosomes = args.chromosomes
    resolution = args.resolution
    matrix_size = args.matrix_size
    output_dir = args.output_dir
    data_dir = args.data_dir
    bigmatrix_dir = args.bigmatrix_dir

    for chr_name in chromosomes:
        positive_name = os.path.join(data_dir, f"positive_chr{chr_name}.txt")
        big_matrix_name = os.path.join(bigmatrix_dir,f"KR_matrix_{resolution}kb.chr{chr_name}")
        np_save_name_positive = os.path.join(output_dir, f"KR_{resolution}kb_matrix_chr{chr_name}_positive.npy")
        get_submatrix_positive(big_matrix_name, positive_name, np_save_name_positive, matrix_size)

    print("Processing complete.")

if __name__ == '__main__':
    main()
