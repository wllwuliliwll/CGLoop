import random
import numpy as np
import argparse
import datetime

def openreadtxt(file_name):
    data = []
    with open(file_name, 'r') as file:
        for row in file:
            tmp_list = row.split('\t')
            tmp_list[-1] = tmp_list[-1].replace('\n', '')
            data.append(tmp_list)
    return data

def count_rows(filename):
    with open(filename, 'r') as f:
        count = sum(1 for _ in f)
    return count

def generate_negative_samples(a, res, matrix_size, row_base_path, big_matrix_base_path, output_base_path):
    for n in range(len(a)):
        row_name = f'{row_base_path}/positive_chr{a[n]}.txt'
        big_matrix_name = f'{big_matrix_base_path}/chr{a[n]}-{res}kb.KRnorm'
        path_out_name = f'{output_base_path}/negative_chr{a[n]}-{res}KB.txt'
        
        row = openreadtxt(row_name)
        end = count_rows(big_matrix_name)
        row_bin = []

        # Parse positive samples
        for i in range(len(row)):
            del row[i][3:5]
            row_bin.append([int(x) for x in row[i][1:]])

        # Distance calculation for positive samples
        key = []
        value = []
        row_bin_loop = []
        for i in range(len(row)):
            distance = row_bin[i][1] - row_bin[i][0]
            if distance not in key:
                key.append(distance)
                value.append(1)
                row_bin_loop.append([])
                row_bin_loop[-1].append(row_bin[i])
            else:
                index = key.index(distance)
                num = value[index]
                value[index] = num + 1
                row_bin_loop[index].append(row_bin[i])

        # Generate negative samples with same distance
        indexbin_same_distance = []
        given_numbers = []
        for i in range(len(row_bin_loop)):
            given_numbers1 = [bin[0] for bin in row_bin_loop[i]]
            given_numbers.append(given_numbers1)

        # Generate random negative samples
        for i in range(len(value)):
            given_numbers2 = given_numbers[i]
            random_generate = []
            for j in range(value[i] * 2):
                random_num = random.randint(1, end - key[i] + 1)
                while random_num in given_numbers2 or random_num in random_generate:
                    random_num = random.randint(1, end - key[i] + 1)
                random_generate.append(random_num)
                indexbin_same_distance.append([random_num, random_num + key[i]])

        # Generate random negative samples from all distances
        alread_existing = row_bin + indexbin_same_distance
        indexbin_random_distance = []
        for i in range(len(row)):
            number_key = random.choice(key)
            numbers = random.randint(1, end - number_key + 1)
            c = [numbers, numbers + number_key]
            while c in alread_existing or c in indexbin_random_distance:
                numbers = random.randint(1, end - number_key + 1)
                c = [numbers, numbers + number_key]
            indexbin_random_distance.append(c)

        # Generate negative samples with greater distance
        indexbin_bigger_distance = []
        lower_bound = 1
        upper_bound = end
        threshold = max(key)
        for i in range(len(row)):
            x1, x2 = random.randint(lower_bound, upper_bound), random.randint(lower_bound, upper_bound)
            dist = abs(x1 - x2)
            while dist <= threshold:
                x1, x2 = random.randint(lower_bound, upper_bound), random.randint(lower_bound, upper_bound)
                dist = abs(x1 - x2)
            if x1 < x2:
                indexbin_bigger_distance.append([x1, x2])
            else:
                indexbin_bigger_distance.append([x2, x1])

        # Combine all generated negative samples
        indexbin = indexbin_same_distance + indexbin_bigger_distance + indexbin_random_distance

        # Write negative samples to file
        with open(path_out_name, 'w+') as f_out:
            for item in indexbin:
                f_out.write('\t'.join(map(str, item)) + '\n')

        print(f"完成第{a[n]}条染色体的负样本生成")
        print(f"正样本共有：{len(row_bin)}")
        print(f"负样本共有：{len(indexbin)}")

def main():
    parser = argparse.ArgumentParser(description="Generate negative samples based on positive samples.")
    parser.add_argument('-a', '--chromosomes', type=str, nargs='+', default=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X'], help="List of chromosomes (e.g., ['1', '2', '3', ...])")
    parser.add_argument('-r', '--resolution', type=int, default=5, help="Resolution for the matrix (e.g., 5, 10, etc.)")
    parser.add_argument('-m', '--matrix_size', type=int, default=21, help="Size of the matrix (e.g., 21)")
    parser.add_argument('-p','--row_base_path', type=str, required=True, help="Base path for positiveTxt sample dir")
    parser.add_argument('-n','--norm_factor',type=str, required=True, help="Base path for norm_factor dir")
    parser.add_argument('-o', '--output_path',type=str, required=True, help="Base path for output negative sample dir")

    args = parser.parse_args()

    # Generate negative samples
    generate_negative_samples(args.chromosomes, args.resolution, args.matrix_size, args.row_base_path, args.norm_factor, args.output_path)

    print(f"Negative sample generation completed at {datetime.datetime.now()}.")

if __name__ == "__main__":
    main()
