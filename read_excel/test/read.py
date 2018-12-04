# -*- coding: utf-8 -*-
import xlrd

xlsfile = r"data.xls"# 打开指定路径中的xls文件

book = xlrd.open_workbook(xlsfile)#得到Excel文件的book对象，实例化对象

sheet0 = book.sheet_by_index(0) # 通过sheet索引获得sheet对象
#sheet1 = book.sheet_by_name(sheet_name)# 通过sheet名字来获取sheet对象，输出等效于sheet0

print ("sheet[0]对象",type(sheet0))
sheet_name = book.sheet_names()[0]# 获得指定索引的sheet表名字
print ("sheet名字",sheet_name)


nrows = sheet0.nrows    # 获取行总数
print ("总行数",nrows)
ncols = sheet0.ncols    #获取列总数
print ("总列数",ncols)

#循环打印每一行的内容
print("循环打印每一行的数据")
for i in range(nrows):
    print (sheet0.row_values(i))
#循环打印每一列的内容
print("循环打印每一列的数据")
for i in range(ncols):
    print (sheet0.col_values(i))

row_data = sheet0.row_values(0)     # 获得第1行的数据列表
print ("获取第一行数据",row_data)
col_data = sheet0.col_values(0)     # 获得第1列的数据列表
print ("获取第一列数据",col_data)
# 通过坐标读取表格中的数据
cell_value1 = sheet0.cell_value(0, 0)
print ("获取[0,0]数据",cell_value1)
cell_value2 = sheet0.cell_value(0, 1)
print ("获取[0,1]数据",cell_value2)
#打印至指定文件
f_o = open('data.csv', 'w', encoding='utf-8')
for i in range(nrows):
    #print (sheet0.row_values(i))
	text = sheet0.row_values(i)
	print("text:",text)
	text_str = str(text).replace(',', '').replace('\'', '').replace('[', '').replace(']', '').replace(' ', '')
	print("text_str:",text_str)
	
	f_o.write(text_str)
	f_o.write('\n')
f_o.close()