# -*- codding: utf-8 -*-
'''
@Author : Yuren
@Dare   : 2022/4/11-2:03 下午
'''
'''
Encode_data/0/xxx.png
Encode_data/……/xxx.png
……


将800类数据集分开存放

该文件在OUT同级目录下执行
'''

import os
import random
import shutil


randseed = 800
SIZE = 80000
random.seed(randseed)
allsample = list(range(SIZE))

if not os.path.exists('Speckle_data'):
    os.mkdir('Speckle_data')
for name in range(800):
    path = 'Speckle_data/'+str(name)
    if not os.path.exists(path):
        os.mkdir(path)


print('Class_image_file create success!!!')


f = open('Class800_80000.txt','r')
Label = []
for line in f.readlines():
    line = line.strip('\n')
    line = line.split('\t')
    Label.append(int(line[0]))
f.close() 
for ind in allsample:
    nowpath = 'Speckle_data/'+str(Label[ind])+'/'+str(ind+1)+'.png'
    imagepath = 'OUT/'+str(ind+1)+'.png'
    if not os.path.exists(imagepath):
        print("NO test image Exist!!!")
        exit(0)
    shutil.move(imagepath,nowpath)
print('Class_image OVER!!!')


