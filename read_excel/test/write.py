#!/usr/bin/env python
# coding=utf-8
#需要xlwt库的支持
#import xlwt
from xlwt import *
#指定file以utf-8的格式打开
file = Workbook(encoding = 'utf-8')
#指定打开的文件名sheet
table = file.add_sheet('data')

#字典数据
data = {
        "1":["张三",150,120,100],
        "2":["李四",90,99,95],
        "3":["王五",60,66,68]
        }

 
ldata = []
#for循环指定取出key值存入num中
num = [a for a in data]
#字典数据取出后，先排序
num.sort()

#for循环将data字典中的键和值分批的保存在ldata中 
for x in num:

    t = [int(x)]
    for a in data[x]:
        t.append(a)
    ldata.append(t)
 
for i,p in enumerate(ldata):
#将数据写入文件,i是enumerate()函数返回的序号数
    for j,q in enumerate(p):
        # print i,j,q
        table.write(i,j,q)
file.save('0712.xls')