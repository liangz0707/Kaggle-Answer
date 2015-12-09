# coding:utf-8
__author__ = 'liangz14'

target = list()
with open("result.csv") as t:
    lines = t.readlines()
    for line in lines[1:]:
        tmp_str = line.split(",")
        target.append(int(tmp_str[1]))


with open("final.csv",'w') as r:
    r.write("ImageId,Label\n")
    for i in range(len(target)):
        r.write("%d,%d\n" % (i+1,target[i]))
