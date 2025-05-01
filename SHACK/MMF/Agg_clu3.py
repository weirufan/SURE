# -*- codding: utf-8 -*-
'''
@Author : Yuren
@Dare   : 2022/6/15-13:43

这个脚本用于从800图片样本中选择256个最佳的样本，进行层次聚类，以期待得到最好的结果的算法。需要消耗巨大的时间成本。

data: 800*100 = 80000张图
label:0-799

首选选取label 0-255的图，进行层次聚类。选择分类精度最低的10个类丢弃，从待选池中抽取10个。更换data和label集，重新聚类，如此反复。

需要定义一个函数来完成两个功能：1。输出精度最低的十个类别的编号   2。根据编号来调整select_label 池
agg_best_dir  agg_match  agg_model
nohup python -u Agg_clu.py --Agg_SIZE 20 >nohup_out_01.out 2>&1 &
python Agg_clu.py
'''

# nohup python -u Agg_clu4.py --Agg_SIZE 100 --path Encode_data --txt_path Class256_25600.txt --Ten_choose > 100class_final_test.out 2>&1 &
# nohup python -u Agg_clu3.py --Ten_choose > nohup_256class_01.out 2>&1 &

# nohup python -u Agg_clu3.py --iter_nums 150 --threshold 0.96 --Ten_choose >noh_03.out 2>&1 &

# nohup python -u Agg_clu3.py --iter_nums 150 --threshold 0.97 --Ten_choose >noh_04.out 2>&1 &


# import numpy as np
# from PIL import Image
# from scipy.optimize import linear_sum_assignment as linear_assignment
# import random


from sklearn.cluster import AgglomerativeClustering
import os
import joblib
import time
import argparse

from agg_tools import *
# Options ----------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--iter_nums", type=int, default=100)
parser.add_argument("--threshold", type=float,default=0.95)
parser.add_argument("--Agg_SIZE", type=int, default=256)
parser.add_argument("--path", type=str,default='Speckle_data')
parser.add_argument("--txt_path", type=str,default='Class800_80000.txt')
parser.add_argument("--Ten_choose",default=False,action="store_true")

config = parser.parse_args()
# -------------------------------------------------------------------------------
iter_nums = config.iter_nums  #set_iter_nums
threshold = config.threshold
Agg_SIZE = config.Agg_SIZE
path = config.path
txt_path = config.txt_path
Ten_choose = config.Ten_choose


# 默认都label池
#select_label = list(range(0,Agg_SIZE))
#wait_select_label = list(range(Agg_SIZE,800))

# temp = np.loadtxt('agg_best_dir/64_best_label_0621.txt')
# temp = temp.astype(np.int32)

# select_label = temp
select_label = random.sample(list(range(800)),Agg_SIZE)
select_label.sort()
wait_select_label = []
for wsl in range(800):
    if wsl in select_label:
        continue
    else:
        wait_select_label.append(wsl)

if not os.path.exists('agg_match'):
    os.mkdir('agg_match')
if not os.path.exists('agg_model'):
    os.mkdir('agg_model')
if not os.path.exists('agg_best_dir'):
    os.mkdir('agg_best_dir')




best_acc = 0 # 记录最佳精度
best_label = [] #记录最佳label组
best_iter = 0 #记录最佳迭代数

#先载入全部label

Label = []
f = open(txt_path, 'r')
for line in f.readlines():
    line = line.strip('\n')
    line = line.split('\t')
    Label.append(int(line[0]))
f.close()
Label = np.array(Label)  #这里是总label池

true_label = read_labels2(select_label)

#图像路径处理
paths = os.listdir(path)
print("label numbers:{}".format(len(paths)))


print('--'*20)
time0 = time.time()
for iter_num in range(iter_nums):
    time1 = time.time()

    print('The {} iter Start!'.format(iter_num))

    print('Start load image')

    # 读取图像数据 np.array
    now_data = read_imgs2(select_label,path)

    print('image load OVER cost time:{:.2f}'.format(time.time()-time1))
    print('image shape:{}'.format(now_data.shape))
    time2 = time.time()
    print('AggClustering Start!!!')
    Agg = AgglomerativeClustering(n_clusters=Agg_SIZE)
    Agg.fit(now_data)
    print('AggClustering over cost time{:.2f}'.format(time.time()-time2))
    pre_label = Agg.labels_

    accuracy ,match ,confuse = acc(true_label,pre_label,Agg_SIZE)

    print('The iter_nums = {} , and Acc:{}'.format(iter_num,accuracy))
    np.savetxt('agg_match/0921/64iter_{}_agg_out_0921.txt'.format(iter_num),match)
    joblib.dump(Agg,'agg_model/0921/64iter_{}_Agg_{:.2f}_0921.pkl'.format(iter_num,accuracy))

    if accuracy > best_acc:
        best_acc = accuracy
        best_label = np.array(select_label)
        best_iter = iter_num
        np.savetxt('agg_best_dir/64_best_label_0921.txt',best_label)
        print('the now best label in Iter: {}'.format(best_iter))
        if accuracy==1:
            print("Task OVER! The ACC == 1.00 !")
            break


    # 这里是调整select_label 和 wait_select_label
    select_label,wait_select_label,ggflag = low_number(confuse,select_label,wait_select_label,threshold,Ten_choose)
    # 调整结束
    print("the Wait_select_label len is {}".format(len(wait_select_label)))

    if ggflag:
        if ggflag == 1:
            if threshold < 1:
                threshold += 0.01
                print("Now the threshold is {}".format(threshold))
                if threshold == 1:
                    print("threshold reached limit!!!")
                    break
        elif ggflag==2:
            print('Iter over because ggflag==2 !!!')
            break
    print('The {} iter Over Cost time {:.2f}'.format(iter_num,time.time()-time1))
    print('--' * 20)

# np.savetxt('agg_best_dir/64iter_{}_label.txt'.format(best_iter),best_label)
print('Iter_nums is {} best_acc:{:.2f} cost time {:.2f}'.format(best_iter,best_acc,time.time()-time0))
