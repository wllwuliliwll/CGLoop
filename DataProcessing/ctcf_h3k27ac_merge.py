import math
import os
import tempfile
def ctcf_h3k27ac(file1, file2, pro_outfile, outfilename):
    file1_info = {}

    with open(file1, 'r') as infile1:
        for line in infile1:
            line = line.strip('\n').strip('\t')
            chromosome, start1, end1, _, start2, end2 = line.split('\t')

            if chromosome not in file1_info:
                file1_info[chromosome] = []
            file1_info[chromosome].append([chromosome, int(start1), int(end1), int(start2), int(end2)])

    for chromosome in file1_info:
        file1_info[chromosome] = sorted(file1_info[chromosome], key=lambda x: x[1])

    merge = []
    with open(file2, 'r') as infile2:
        for line in infile2:
            line = line.strip('\n').strip('\t')
            chromosome, start1, end1, _, start2, end2 = line.split('\t')

            chromosome_info = file1_info.get(chromosome)
            if chromosome_info is None:
                print(f"Warning: Chromosome {chromosome} not found. Skipping this line.")
                continue
            
            start1 = int(start1)
            start2 = int(start2)
            end1 = int(end1)
            end2 = int(end2)
        
            file1_info_temp = file1_info[chromosome]
            file1_info_temp = sorted(file1_info_temp, key=lambda x: x[1])

            # 使用二分查找法进行查找和合并
            left, right = 0, len(file1_info_temp) - 1
            position = -1
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

            left, right = 0, len(file1_info_temp) - 1
            position = -1
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

            left, right = 0, len(file1_info_temp) - 1
            position = -1
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
            left, right = 0, len(file1_info_temp) - 1
            position = -1
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

            # 如果找到了合适的匹配，加入到 merge 中
            if len(file1_info_temp) == 0:
                merge.append([chromosome, str(start1), str(end1), str(chromosome), str(start2), str(end2)])
    print(len(merge))
    outfile1 = os.path.join(os.path.dirname(pro_outfile), outfilename+'.bedpe')
    with open(outfile1, 'w+') as out:
        for info in merge:
            temp = '\t'.join(info)
            out.write(temp + '\n')

    #print(f"merge results saved in: {outfile1}")
    return outfile1


def merge(file1,file2,pro_outfile,outfilename):
    #print(file1)
    file1_info = {}
    infile = open(file1, 'r')
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
    #print(file2)
    infile = open(file2, 'r')
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

    outfile_path = os.path.join(os.path.dirname(pro_outfile), outfilename + '.bedpe')
    #print(outfile_path)
    outfile2 = open(outfile_path,'w+')
    n = 0
    for chromosome in file1_info:
        for info in file1_info[chromosome]:
            # print(info)
            n += 1
            temp = '\t'.join(info)
            outfile2.write(temp + '\n')
    outfile2.close()
    print(n)  # 打印行数
    return outfile_path  

def apart_merge(file, pro_outfile, outfilename):
    if isinstance(file, str):
        with open(file, 'r') as infile:
            res = 5000
            clist = []
            for line in infile:
                line = line.strip('\n')
                line = line.strip('\t')
                chromosome, start1, end1, _, start2, end2 = line.split('\t')
                s1 = int(start1) // res
                e1 = int(end1) // res
                s2 = int(start2) // res
                e2 = int(end2) // res
                for i in range(s1 + 1, e1 + 1):
                    for j in range(s2 + 1, e2 + 1):
                        clist.append([chromosome, i, j])

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

    outfile_path0 = os.path.join(os.path.dirname(pro_outfile), outfilename + '.bedpe')
    with open(outfile_path0, 'w+') as outfile0:
        n = 0
        for info in new_lst:
            n += 1
            info[1] = str(info[1])
            info[2] = str(info[2])
            temp = '\t'.join(info)
            outfile0.write(temp + '\n')

    #print(f"deal with {n} ")
    return outfile_path0


