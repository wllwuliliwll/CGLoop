import random
import numpy as np
def openreadtxt(file_name):
    data = []
    file = open(file_name,'r')
    #print('file_data=')
    #print(file_data)
    for row in file:
        tmp_list = row.split(' ')
        tmp_list = row.split('\t')
        tmp_list[-1] = tmp_list[-1].replace('\n','')
        data.append(tmp_list)
    return data

def count_rows(filename):
    with open(filename, 'r') as f:
        count = 0
        for line in f:
            count += 1
        return count

a = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']
res=5
matrix_size=21
for n in range(len(a)):
    row_name = '/path/of/positive_centerpoint.txt'
    big_matrix_name = '/path/of/bigmatrix.chr'
    path_out_name = 'path/of/negative_centerpoint.txt'
    all = [] #生成的小np矩阵
    row = openreadtxt(row_name)
    end = count_rows(big_matrix_name)
    row_bin = [] 
    for i in range(len(row)):
        del row[i][3:5]
        row_bin.append([int(x) for x in row[i][1:]])


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

# 随机产生二倍的负样本 相同距离  
    indexbin_same_distance = []
    given_numbers = []
    for i in range(len(row_bin_loop)):
        given_numbers1 = []
        for j in range(len(row_bin_loop[i])):
            given_numbers1.append(row_bin_loop[i][j][0])
        given_numbers.append(given_numbers1)

    #随机产生一些整数
    for i in range(len(value)):
        given_numbers2 = given_numbers[i] 
        random_generate = [] 
        for j in range(value[i] * 2): 
            random_num = random.randint(1, end-key[i]+1)
        
            while random_num in given_numbers2 or random_num in random_generate:
                random_num = random.randint(1, end-key[i]+1)
            random_generate.append(random_num)
            indexbin_same_distance.append([random_num, random_num + key[i]])

#随机的从所有距离中，一倍数量 
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

#随机产生 大距离，一倍
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
    indexbin = indexbin_same_distance + indexbin_bigger_distance +indexbin_random_distance


    path_out = path_out_name
    t = ''
    with open(path_out, 'w+') as f_out:
        for i in range(len(indexbin)):
            for j in range(len(indexbin[i])):
                f_out.write(str(indexbin[i][j]))
                f_out.write('\t')
                #f_out.write('chr' + str(a[n]) + '\t')
            f_out.write('\n')

    print("这是第%s条染色体已完成" % (str(a[n])))
    print("正样本共有：%d" % (len(row_bin)))
    print("负样本共有：%d" % (len(indexbin)))
now = datetime.datetime.now()
print("Current time is:", now)