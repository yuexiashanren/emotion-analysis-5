# -*- coding: utf-8 -*-
import xlrd

xlsfile = r"实验练习.xls"# 打开指定路径中的xls文件

book = xlrd.open_workbook(xlsfile)#得到Excel文件的book对象，实例化对象

sheet0 = book.sheet_by_index(0) # 通过sheet索引获得sheet对象
#sheet1 = book.sheet_by_name(sheet_name)# 通过sheet名字来获取sheet对象，输出等效于sheet0

print ("sheet[0]类别",type(sheet0))
sheet_name = book.sheet_names()[0]# 获得指定索引的sheet表名字
print ("sheet[0]名字",sheet_name)


nrows = sheet0.nrows    # 获取行总数
print ("总行数",nrows)
ncols = sheet0.ncols    #获取列总数
print ("总列数",ncols)
#float,16.0
#print ("type[,5]",type(sheet0.cell_value(1,5)),sheet0.cell_value(1,5))
#str,...
#print ("type[,5]",type(sheet0.cell_value(1,6)),sheet0.cell_value(1,6))
j = 0
f_o = open('0716.csv', 'w', encoding='utf-8')
for i in range(nrows):
    #print (sheet0.row_values(i))
	choose = sheet0.cell_value(i,5)
	if(choose == 16):
		j += 1
		text_str = sheet0.cell_value(i,6)
				
		f_o.write(text_str)
		f_o.write('\n')
	
f_o.close()
print("累计语料",j)

