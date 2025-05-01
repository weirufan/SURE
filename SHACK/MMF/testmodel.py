# -*- codding: utf-8 -*-
'''
@Author : Yuren
@Dare   : 2022/6/20-13:35
'''
from sklearn.cluster import AgglomerativeClustering
import os
import joblib
import time
import argparse

from agg_tools import *


def transf(y_pred, match):
    new_data = np.zeros(len(y_pred), dtype=np.int64)
    for i in range(len(y_pred)):
        new_data[i] = match[1, y_pred[i]]
    return new_data


path = 'Speckle_data'
select_label = np.loadtxt('agg_best_dir/64_best_label.txt')
select_label = select_label.astype(np.int16)
print(len(select_label))

match = np.loadtxt('agg_match/64iter_98_agg_out.txt')
match = match.astype(np.int16)


true_label = read_labels2(select_label)

now_data = read_imgs2(select_label,path)

agg_model = joblib.load('agg_model/64iter_98_Agg_0.97_.pkl')

pre_label = agg_model.labels_
pre_label = transf(pre_label,match)
print(np.sum(pre_label==true_label)/len(pre_label))
