#!/usr/bin/env python
# coding: utf-8
import math
import argparse
import os
import tempfile
import ctcf_h3k27ac_merge  # 导入ctcf_h3k27ac_merge模块

def generate_temp_file(prefix, suffix=".bedpe"):
    """
    生成临时文件路径
    :param prefix: 文件名前缀
    :param suffix: 文件名后缀
    :return: 临时文件路径
    """
    temp_dir = tempfile.gettempdir()  # 获取系统临时目录
    return os.path.join(temp_dir, f"{prefix}{suffix}")

def ctcf_h3k27ac_norm(ctcffile,h3k27acfile,pro_outdir):#.bedpe
    filepath = ctcffile
    ctcfbin = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    ctcfchr = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12',
                'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',  'chr20', 'chr21', 'chr22', 'chrX']
    lower = 30000
    highter = 3000000
    with open(filepath) as f:
        for line in f:
            line = line.split()
            if line[0] not in ctcfchr:
                print(f"Warning: Chromosome {line[0]} not found in ctcfchr. Skipping line.")
                continue
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

    ctcfsavefile = os.path.join(os.path.dirname(pro_outdir), 'ctcf.bedpe')
    outfile = open(ctcfsavefile, 'w+')
    for info in final1:
        temp = '\t'.join(info)
        outfile.write(temp + '\n')
    outfile.close()

    ctcfbin1 = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    ctcfchr1 = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12',
                'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',  'chr20', 'chr21', 'chr22', 'chrX']


    filepath1 = h3k27acfile
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
    h3k27acsavefile = os.path.join(os.path.dirname(pro_outdir), 'h3k27ac.bedpe')
    outfile = open(h3k27acsavefile, 'w+')
    for info in final2:
        temp = '\t'.join(info)
        outfile.write(temp + '\n')
    outfile.close()
    return ctcfsavefile,h3k27acsavefile 


def merge_positive(infile,outfile_dir):
    '''将合并的正样本分成不同染色体不同文件的正样本'''
    a = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12',
        'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']
    new_list = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    infile = open(infile, 'r')
    for line in infile:
        line = line.strip('\n')
        line = line.strip('\t')
        chromosome, bin1, bin2= line.split('\t')
        chr_index = a.index(chromosome)
        new_list[chr_index].append([chromosome, bin1, bin2])

    num = 0
    for i in range(len(new_list)):
        outfile_path = os.path.join(outfile_dir, f'positive_{a[i]}.txt')
        with open(outfile_path, 'w') as outfile:
            n = 0  
            for info in new_list[i]:
                n += 1
                num += 1
                info[1] = str(info[1])
                info[2] = str(info[2])
                temp = '\t'.join(info)  
                outfile.write(temp + '\n')  
            print(f"正样本第{a[i]}染色体，共有正样本：{n}个")
        
    print(f"共有{num}个正样本")

import io
def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="Simplified Process for CTCF and H3K27AC files")
    parser.add_argument('-i1', '--ctcf_file', type=str, required=True, help='Path to CTCF file')
    parser.add_argument('-i2', '--h3k27ac_file', type=str, required=True, help='Path to H3K27AC file')
    parser.add_argument('-o', '--positive_file', type=str, required=True, help='Path to save positive result file')
    parser.add_argument('-p', '--pro_outfile', type=str, required=True, help='Path to save process data dir')

    # 解析命令行参数
    args = parser.parse_args()

    # 动态生成中间文件路径
    #del_ctcf_path = generate_temp_file("del_ctcf")
    #del_h3k27ac_path = generate_temp_file("del_h3k27ac")
    #merge_path = generate_temp_file("merge")
    #apart_merge_path = generate_temp_file("apart_merge")

    # print(f"中间文件将存储在：\n"
    #       f"del_ctcf_path: {del_ctcf_path}\n"
    #       f"del_h3k27ac_path: {del_h3k27ac_path}\n"
    #       f"merge_path: {merge_path}\n"
    #       f"apart_merge_path: {apart_merge_path}\n")

    # 调用相关函数
    file1, file2 = ctcf_h3k27ac_norm(args.ctcf_file, args.h3k27ac_file, args.pro_outfile)
    print("file1",file1)
    print("file2",file2)
    del_h3k27ac_path = ctcf_h3k27ac_merge.ctcf_h3k27ac(file1, file2, args.pro_outfile, 'del_h3k27ac')
    print("del_h3k27ac_path",del_h3k27ac_path)
    del_ctcf_path = ctcf_h3k27ac_merge.ctcf_h3k27ac(del_h3k27ac_path, file1, args.pro_outfile, 'del_ctcf')
    print("del_ctcf_path",del_ctcf_path)
    ###############################
    print("del_ctcf_path",del_ctcf_path)
    merge_path = ctcf_h3k27ac_merge.merge(del_ctcf_path, del_h3k27ac_path, args.pro_outfile, 'merge')
    print("merge_path",merge_path)
    apart_merge_path = ctcf_h3k27ac_merge.apart_merge(merge_path, args.pro_outfile, 'apart_merge')
    print("apart_merge_path",apart_merge_path)
    # 调用正样本生成逻辑
    merge_positive(apart_merge_path, args.positive_file)

    print(f"最终正样本文件已保存到：{args.positive_file}")


    
if __name__ == "__main__":
    main()




    # #查看bedpe文件格式是否正确，不正确需要处理，我这里是做了取整操作
    # ctcf_file = 'path/of/ctcf_chiapet.bedpe'
    # h3k27ac_file ='path/of/h3k27ac_hichip.bedpe'
    # del_ctcf = 'path/of/del_ctcf.bedpe'
    # del_h3k27ac = 'path/of/del_h3k27ac.bedpe'
    # merge = 'path/of/merge.bedpe'
    # file1,file2 = ctcf_h3k27ac_norm(ctcf_file,h3k27ac_file)
    # ctcf_h3k27ac_merge.ctcf_h3k27ac(file1,file2,del_h3k27ac)
    # ctcf_h3k27ac_merge.ctcf_h3k27ac(del_h3k27ac,ctcf_file,del_ctcf)
    # ctcf_h3k27ac_merge.merge(del_h3k27ac,del_h3k27ac,merge)
    # ctcf_h3k27ac_merge.apart_merge(merge,apart-merge)

