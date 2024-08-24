#!/usr/bin/env python
# coding: utf-8
import math

filepath = "/path/of/ctcf-chiapet.bedpe"
ctcfbin = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
ctcfchr = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12',
            'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',  'chr20', 'chr21', 'chr22', 'chrX']
lower = 30000
highter = 3000000
with open(filepath) as f:
    for line in f:
        line = line.split()
        index = ctcfchr.index(line[0])

        updated_number_left1 = str(math.floor(int(line[1]) / 5000) * 5000)
        updated_number_left2 = str(math.ceil(int(line[2]) / 5000) * 5000)
        updated_number_right1 = str(math.floor(int(line[4]) / 5000) * 5000)
        updated_number_right2 = str(math.ceil(int(line[5]) / 5000) * 5000)
        if int(updated_number_right1) - int(updated_number_left1) < lower or int(updated_number_right1) - int(updated_number_left1) > highter:
            continue
        else:
            ctcfbin[index].append([line[0],updated_number_left1,updated_number_left2,line[3],updated_number_right1,updated_number_right2])

new_ctcf1 = []
for i in range(len(ctcfbin)):
    tuple_list = []
    for item in ctcfbin[i]: 
        tuple_item = tuple(item)
        tuple_list.append(tuple_item)
    new_tuple_list = list(set(tuple_list)) 
    new_list = [list(item) for item in new_tuple_list] 
    new_ctcf1.append(new_list)

new1 = []

def sort_2d_list(lst, col):
    return sorted(lst, key=lambda x: (int(x[col]), int(x[col+3])))
for i in range(len(new_ctcf1)):
    lst_sorted = sort_2d_list(new_ctcf1[i], 1)
    new1.append(lst_sorted)


final1 = []
for i in range(len(new1)):
    for j in range(len(new1[i])):
        final1.append(new1[i][j])
print(len(final1))

outfile = open("/path/of/ctcf.bedpe", 'w+')
for info in final1:
    temp = '\t'.join(info)
    outfile.write(temp + '\n')
outfile.close()

ctcfbin1 = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
ctcfchr1 = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12',
            'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',  'chr20', 'chr21', 'chr22', 'chrX']


filepath1 = '/path/of/h3k27ac-hichip.bedpe'
with open(filepath1) as f:
    for line in f:
        line = line.split()
        index = ctcfchr1.index(line[0])
        if int(line[4]) - int(line[1]) < lower or int(line[4]) - int(line[1]) > 3000000:
            continue
        else:
            ctcfbin1[index].append(line)

new_ctcf2 = []
for i in range(len(ctcfbin1)):
    tuple_list = []
    for item in ctcfbin1[i]:
        tuple_item = tuple(item)
        tuple_list.append(tuple_item)
    new_tuple_list = list(set(tuple_list)) 
    new_list = [list(item) for item in new_tuple_list] 
    new_ctcf2.append(new_list)

new2 = []
for i in range(len(new_ctcf2)):
    lst_sorted = sort_2d_list(new_ctcf2[i], 1)
    new2.append(lst_sorted)

final2 = []
for i in range(len(new2)):
    for j in range(len(new2[i])):
        final2.append(new2[i][j])
print(len(final2))

outfile = open('/path/of/h3k27ac.bedpe', 'w+')
for info in final2:
    temp = '\t'.join(info)
    outfile.write(temp + '\n')
outfile.close()

'''去除嵌套'''
file1_info = {}
#infile1 = open('/path/of/ctcf.bedpe', 'r')#【1】ctcf.bedpe+h3k27ac.bedpe=delete_h3k27ac.bedpe
infile1 = open('/path/of/delete_h3k27ac.bedpe', 'r')#【2】delete_h3k27ac.bedpe+ctcf.bedpe=delete_ctcf.bedpe
for line in infile1:
    line = line.strip('\n')
    line = line.strip('\t')
    chromosome, start1, end1, _, start2, end2 = line.split('\t')

    if chromosome not in file1_info:
        file1_info[chromosome] = []
    file1_info[chromosome].append([chromosome, int(start1), int(end1), int(start2), int(end2)])
infile1.close()

for file1_info_chromosome in file1_info:
    file1_info[file1_info_chromosome] = sorted(file1_info[file1_info_chromosome], key=lambda x: x[1])

infile1.close()

merge = []
#infile2 = open('/path/of/h3k27ac.bedpe', 'r')#【1】
infile2 = open('/path/of/ctcf.bedpe', 'r')#【2】
for line in infile2:
    line = line.strip('\n')
    line = line.strip('\t')
    chromosome, start1, end1, _, start2, end2 = line.split('\t')
    start1 = int(start1)
    start2 = int(start2)
    end1 = int(end1)
    end2 = int(end2)
 
    file1_info_temp = file1_info[chromosome]
    file1_num = len(file1_info_temp)
    file1_info_temp = sorted(file1_info_temp, key=lambda x: x[1])

    left = 0
    right = file1_num - 1
    position = -1
    delete_flag = -1
    while left <= right:
        middle = (left + right) // 2
        temp = file1_info_temp[middle][1]
        if temp > start1:
            right = middle - 1
        elif temp < start1:
            left = middle + 1
        else:
            position = middle
            file1_info_temp = sorted(file1_info_temp[:position + 1], key=lambda x: x[2]) 
            break
    if position == -1:
        position = right + 1
        file1_info_temp = sorted(file1_info_temp[:position], key=lambda x: x[2])
    file1_num = len(file1_info_temp)

    left = 0
    right = file1_num - 1
    position = -1
    delete_flag = -1
    while left <= right:
        middle = (left + right) // 2
        temp = file1_info_temp[middle][2]
        if temp > end1:
            right = middle - 1
        elif temp < end1:
            left = middle + 1
        else:
            position = middle
            break
    if position == -1:
        position = right + 1
    file1_info_temp = sorted(file1_info_temp[position:], key=lambda x: x[3])
    file1_num = len(file1_info_temp)

    left = 0
    right = file1_num - 1
    position = -1
    delete_flag = -1
    while left <= right:
        middle = (left + right) // 2
        temp = file1_info_temp[middle][3]
        if temp > start2:
            right = middle - 1
        elif temp < start2:
            left = middle + 1
        else:
            position = middle
            file1_info_temp = file1_info_temp[:position + 1]
            break
    if position == -1:
        position = right + 1
        file1_info_temp = file1_info_temp[:position]


    file1_info_temp = sorted(file1_info_temp[position:], key=lambda x: x[4])
    file1_num = len(file1_info_temp)
    left = 0
    right = file1_num - 1
    position = -1
    delete_flag = -1
    while left <= right:
        middle = (left + right) // 2
        temp = file1_info_temp[middle][4]
        if temp > end2:
            right = middle - 1
        elif temp < end2:
            left = middle + 1
        else:
            position = middle
            break
    if position == -1:
        position = right + 1
    file1_info_temp = sorted(file1_info_temp[position:], key=lambda x: x[3])


    if len(file1_info_temp) == 0:
        merge.append([chromosome, str(start1), str(end1), str(chromosome), str(start2), str(end2)])
infile2.close()
print(len(merge))
#outfile = open("/path/of/delete_h3k27ac.bedpe", 'w+')#【1】
outfile = open("/path/of/delete_ctcf.bedpe", 'w+')#【2】
for info in merge:
    temp = '\t'.join(info)
    outfile.write(temp + '\n')
outfile.close()


'''两个文件进行合并'''
file1_info = {}
infile = open('/path/of/delete_ctcf.bedpe', 'r')
for line in infile:
    line = line.strip('\n')
    line = line.strip('\t')
    line = line.split('\t')
    if len(line) == 0:
        continue
    chromosome = line[0]
    if chromosome not in file1_info:
        file1_info[chromosome] = []
    file1_info[chromosome].append(line)
infile.close()

infile = open('/path/of/delete_h3k27ac.bedpe', 'r')
for line in infile:
    line = line.strip('\n')
    line = line.strip('\t')
    line = line.split('\t')
    if len(line) == 0:
        continue
    chromosome = line[0]
    if chromosome not in file1_info:
        file1_info[chromosome] = []
    file1_info[chromosome].append(line)
infile.close()

for file1_info_chromosome in file1_info:
    file1_info[file1_info_chromosome] = sorted(file1_info[file1_info_chromosome], key=lambda x: int(x[1]))

outfile = open("/path/of/merge.bedpe", 'w+')
n = 0
for chromosome in file1_info:
    for info in file1_info[chromosome]:
        # print(info)
        n += 1
        temp = '\t'.join(info)
        outfile.write(temp + '\n')
outfile.close()
print(n)

infile = open("/path/of/merge.bedpe", 'r')
res = 10000
clist = []
for line in infile:
    line = line.strip('\n')
    line = line.strip('\t')
    chromosome, start1, end1, _, start2, end2 = line.split('\t')
    s1 = int(start1) // res
    e1 = int(end1) // res
    s2 = int(start2) // res
    e2 = int(end2) // res
    #print(s1,e1,s2,e2)
    #print(end1,e1,end2,e2)
    for i in range(s1+1 , e1 +2):
        for j in range(s2+1 , e2 + 2):
            #print([chromosome, i, j])
            clist.append([chromosome, i, j])
            #print(clist)
            


def remove_duplicates(lst):
    seen = {}
    result = []
    for item in lst:
        key = tuple(item)
        if key not in seen:
            seen[key] = True
            result.append(item)
    return result
new_lst = remove_duplicates(clist)

outfile = open("/path/of/apart-merge.bedpe", 'w+')
n = 0
for info in new_lst:
    n += 1
    info[1] = str(info[1])
    info[2] = str(info[2])
    temp = '\t'.join(info)
    outfile.write(temp + '\n')
outfile.close()
print(n)

'''将合并的正样本分成不同染色体不同文件的正样本'''
a = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12',
     'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']
new_list = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
infile = open("/path/of/apart-merge.bedpe", 'r')
for line in infile:
    line = line.strip('\n')
    line = line.strip('\t')
    chromosome, bin1, bin2= line.split('\t')
    chr_index = a.index(chromosome)
    new_list[chr_index].append([chromosome, bin1, bin2])

num = 0
for i in range(len(new_list)):
    outfile = open('/path/of/positive_centerpoint.txt','w+')
    n = 0
    for info in new_list[i]:
        n += 1
        num += 1
        info[1] = str(info[1])
        info[2] = str(info[2])
        temp = '\t'.join(info)
        outfile.write(temp + '\n')
    outfile.close()
    print("正样本第%s染色体，共有正样本：%d个"%(a[i], n))
print("共有%d个正样本"% num)
