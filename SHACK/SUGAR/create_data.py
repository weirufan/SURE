# -*- codding: utf-8 -*-                                                                  
'''
@Author : Yuren
@Dare   : 2022-04-28
'''
import os
import shutil

import random
'''
Label = []
for i in range(2500):
    if i <500:
        Label.append(0)
    elif i < 1000:
        Label.append(1)
    elif i < 1500:
        Label.append(2)
    elif i < 2000:
        Label.append(3)
    else :
        Label.append(4)
f = open('Label5.txt','a')
for i in range(len(Label)):
    f.write(str(Label[i]))
    if i != len(Label)-1:
        f.write('\n')
    else:
        continue
f.close

exit(0)
'''

pathwh = 'wh'
pathwl = 'wl'
pathwo = 'wo'
pathtestl = 'tl'
pathtesth = 'th'
random.seed(905)

wh_path = os.listdir(pathwh)
wl_path = os.listdir(pathwl)
wo_path = os.listdir(pathwo)
tl_path = os.listdir(pathtestl)
th_path = os.listdir(pathtesth)

wh_path.sort(key =lambda x:int(x[:-4]))
wl_path.sort(key = lambda x:int(x[:-4]))
wo_path.sort(key = lambda x:int(x[:-4]))
tl_path.sort(key = lambda x:int(x[:-4]))
th_path.sort(key = lambda x:int(x[:-4]))
for i in range(500):
    wh_path[i]='wh/'+wh_path[i]
    wl_path[i]='wl/'+wl_path[i]
    wo_path[i]='wo/'+wo_path[i]
    tl_path[i]='tl/'+tl_path[i]
    th_path[i]='th/'+th_path[i]




#print(w_path)
#print(wo_path)

all_img = wo_path+wl_path+wh_path+tl_path+th_path
#random.shuffle(all_img)
print(all_img)

print(len(all_img))
'''
Label = []

for i in range(len(all_img)):
    if all_img[i][:2]=='wo':
        Label.append(0)
    elif all_img[i][:2]=='wl':
        Label.append(1)
    elif all_img[i][:2]=='tl':
        Label.append(1)
    else:
        Label.append(2)

print(Label)

f = open('Label3.txt','a')
for i in range(len(Label)):
    f.write(str(Label[i]))
    if i != len(Label)-1:
        f.write('\n')
    else:
        continue
f.close
'''
if not os.path.exists('OUT'):
    os.mkdir('OUT')

for ind in range(len(all_img)):
    nowpath = all_img[ind]
    targetpath = 'OUT/'+str(ind+1)+'.png'
    if not os.path.exists(nowpath):
        print("No image exist")
        exit(0)
    print(targetpath)
    shutil.move(nowpath,targetpath)



