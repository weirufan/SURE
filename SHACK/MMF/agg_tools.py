# -*- codding: utf-8 -*-
'''
@Author : Yuren
@Dare   : 2022/6/15-13:18
'''
import random
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from PIL import Image
import os

def low_number(confuse,select_label:list,wait_select_label:list,threshold = 0.95,Ten_choose = False):
    '''
    :param confuse:             传入混淆矩阵，用于计算精度低于阈值的N个头的位置
    :param select_label:        传入被选择的label列表
    :param wait_select_label:   传入待选择的label列表，拿出不放回的机制
    :param threshold:           精度阈值，低于则被选择
    :param Ten_choose:          是否打开低于阈值的10个头的迭代选项
    :return:                    返回被调整后的 select_label  wait_select_label ggflag(用于判别是否是待选池容量不够导致循环结束)
    '''

    ggflag = 0
    dia_acc = np.diagonal(confuse)/100
    # 考虑到最小10个，比较不稳定，这里设计成精度不到0.95的所有类都重新选取
    low_ind = []
    for i in range(len(select_label)):
        if dia_acc[i] < threshold:
            low_ind.append(i)
    samp_SIZE = len(low_ind)
    if samp_SIZE==0:
        ggflag = 1
        print("samp_SIZE == 0,that is great !!! and threshold will be raised !")
        temp_label2 = []
        for i in range(0,800):
            if i in select_label:
                continue
            else:
                temp_label2.append(i)

        return select_label, temp_label2, ggflag

    print('{} are not OK !'.format(samp_SIZE))
    if Ten_choose:
        if samp_SIZE > 10:
            samp_SIZE = 10
            low_ind = random.sample(low_ind,samp_SIZE)
    low_label = []
    for li in low_ind:
        low_label.append(select_label[li])

    if samp_SIZE > len(wait_select_label):
        print("wait_select_label smaller than samp_SIZE !!!")
        ggflag = 2
        return select_label,wait_select_label,ggflag


    change_label = random.sample(wait_select_label,samp_SIZE)
    # 删除select_label中的low_ind 并且添加change_label
    temp_label = []
    for ll in select_label:
        if ll in low_label:
            continue
        else:
            temp_label.append(ll)
    for ll in change_label:
        temp_label.append(ll)
    temp_label.sort()
    temp_label2 = []
    for ll in wait_select_label:
        if ll in change_label:
            continue
        else:
            temp_label2.append(ll)
    temp_label2.sort()
    return temp_label,temp_label2,ggflag

def read_labels(select_label,Label):
    now_Label = []
    img_index = []
    index = 0

    for nl in Label:
        if nl in select_label:
            id = select_label.index(nl)
            now_Label.append(id)
            img_index.append(index)
        index += 1
    return img_index,np.array(now_Label)

def read_labels2(select_label):
    # now_label = []
    # for lab in select_label:
    #     for i in range(100):
    #         now_label.append(lab)
    # return np.array(now_label)
    now_label = []
    for i in range(len(select_label)):
        for j in range(100):
            now_label.append(i)
    return np.array(now_label)

def read_imgs(path,img_path,img_index):
    now_data = []
    for i in img_index:
        im = Image.open(path+'/'+img_path[i])
        im = np.array(im)
        im = im[64:192,64:192]
        im = im.reshape(-1)
        now_data.append(im)
    return np.array(now_data)
def read_imgs2(select_label,path='Speckle_data'):
    now_data = []
    for img_l in select_label:
        now_p = os.listdir(path+'/'+str(img_l))
        for img_p in now_p:
            im = Image.open(path+'/'+str(img_l)+'/'+img_p)
            im = im.resize((64, 64), Image.NEAREST)
            im = np.array(im)
            # im = im[64:192, 64:192]
            im = im.reshape(-1)
            now_data.append(im)

    return np.array(now_data)

def acc(y_true, y_pred,Agg_SIZE = 256):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = Agg_SIZE
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    match  = linear_assignment(w.max() - w)
    a , b = match
    confuse = np.zeros((D, D), dtype=np.int64)
    suma = 0
    for i in range(len(a)):
        suma += w[a[i],b[i]]
        confuse[b[i], :] = w[a[i], :]
    return suma*1.0/y_pred.size,match,confuse
