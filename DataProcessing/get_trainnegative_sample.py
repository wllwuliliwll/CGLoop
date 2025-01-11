import numpy as np
import datetime
import time
import argparse

def get_submatrix_negative(matrix_input_file_path, center_point_input_file_path, output_file_path, negative_name_sort, interaction_frequency_path, chromosome, matrix_size, res):
    now = datetime.datetime.now()
    print(center_point_input_file_path.split('/')[-1], "start", "    Current time is:", now)
    
    # Read all matrix file to calculate the total number of rows
    with open(matrix_input_file_path, 'r') as all_matrix_file:
        number_all_matrix_file = sum(1 for line in all_matrix_file)

    # Read center point file
    with open(center_point_input_file_path, 'r') as center_point_file:
        point_list = []

        # Read interaction frequency file
        center_point_interaction_frequency_path = {}
        with open(interaction_frequency_path, 'r') as infile:
            for line in infile:
                line = line.strip('\n').split('\t')
                center_point_interaction_frequency_path[str(int(int(line[0]) / (res * 1000)) + 1) + ',' + str(int(int(line[1]) / (res * 1000)) + 1)] = float(line[2])

        num_center_point_all = 0
        num_center_point_delete = 0

        for line_center_point_file in center_point_file:
            num_center_point_all += 1
            temp = line_center_point_file.strip('\n').split('\t')
            temp_center_point = temp[0] + ',' + temp[1]
            if temp_center_point in center_point_interaction_frequency_path and center_point_interaction_frequency_path[temp_center_point] > 1:
                point_list.append([int(temp[0]), int(temp[1])])
                num_center_point_delete += 1

        print("filtered before", num_center_point_all, "，filtered after", num_center_point_delete)
        point_list_temp = sorted(point_list)

        # Split into 100 parts
        num_parts = 100
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

            current_row = 0
            with open(matrix_input_file_path, 'r') as all_matrix_file:
                for line_all_matrix_file in all_matrix_file:
                    current_row += 1
                    # if current_row % 100 == 0:
                    #     print("负样本 ：", center_point_input_file_path.split('/')[-1], "第", point_list_part, "部分已经进行了", (current_row * 100) / number_all_matrix_file, "%")
                    line = line_all_matrix_file.split('\t')
                    for num in range(number_point):
                        rows_point = point_list[num][0]
                        columns_point = point_list[num][1]
                        start_row = rows_point - (matrix_size // 2)
                        end_row = rows_point + (matrix_size // 2)
                        start_column = columns_point - (matrix_size // 2)
                        end_column = columns_point + (matrix_size // 2)
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

            print("raw matrices：", len(all_matrix))
            new_number_point = len(point_list_new)
            new_number_point_all += new_number_point
            print("filtered：", delete_num, "个")
            print("negative matrices：", len(all_matrix_new), "个")
            print("negative point：", len(point_list_new), "个")

            with open(negative_name_sort[:-4] + '_' + str(point_list_part) + '.txt', 'w+') as output_file_point_list:
                for num in point_list_new:
                    output_file_point_list.write('chr' + chromosome + '\t' + str(num[0]) + '\t' + str(num[1]) + '\n')

            all_matrix_new = np.array(all_matrix_new)
            all_matrix_new = all_matrix_new.astype('float32')

            all_matrix_new = all_matrix_new.reshape(-1, matrix_size * matrix_size)
            label_1 = np.zeros((new_number_point, 1))
            label_1 = label_1.astype('float32')
            all_all = np.concatenate((all_matrix_new, label_1), axis=1)
            np.save(output_file_path[:-4] + "_" + str(point_list_part) + '.npy', all_all)

    now = datetime.datetime.now()
    print(center_point_input_file_path.split('/')[-1], "end", " Current time is:", now)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Negative Submatrix for Hi-C Data")
    parser.add_argument('-r', '--res', type=int, default=5, help="Resolution of the data")
    parser.add_argument('-m', '--matrix_size', type=int, default=21, help="Matrix size")
    parser.add_argument('-c', '--chromosomes', type=str, nargs='+', default=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X'], help="List of chromosomes (e.g. 1 2 3 4 X)")
    parser.add_argument('-x', '--matrix_input_file_path', type=str, required=True, help="Path to the hic-matrix input dir")
    parser.add_argument('-p', '--center_point_input_file_path', type=str, required=True, help="Path to the negative center point input dir")
    parser.add_argument('-o', '--output_file_path', type=str, required=True, help="Path to output dir")
    parser.add_argument('-n', '--negative_name_sort', type=str, required=True, help="Path to sorted negative out dir")
    parser.add_argument('-i', '--interaction_frequency_path', type=str, required=True, help="Path to interaction frequency dir")
    args = parser.parse_args()
    # Loop over chromosomes as in the original code
    for chromosome in args.chromosomes:
        point_name = f'{args.center_point_input_file_path}/negative_chr{chromosome}-{args.res}KB.txt'
        big_matrix_name = f'{args.matrix_input_file_path}/KR_matrix_{args.res}kb.chr{chromosome}'  # Using the provided matrix input file path
        np_save_name = f'{args.output_file_path}/KR_{args.res}kb_matrix_chr{chromosome}_negative.npy'
        interaction_frequency_path = f'{args.interaction_frequency_path}/chr{chromosome}-{args.res}kb.KRobserved'
        negative_center_delete_sort = f'{args.negative_name_sort}negative_chr{chromosome}-{args.res}KB-sort.txt'
        get_submatrix_negative(big_matrix_name, point_name, np_save_name, negative_center_delete_sort, interaction_frequency_path, chromosome, args.matrix_size, args.res)
