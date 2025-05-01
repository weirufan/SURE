# -*- codding: utf-8 -*-
'''
@Author : Yuren
@Dare   : 2022/3/30-1:38 下午

'''

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import os
from scipy.optimize import linear_sum_assignment as linear_assignment
import joblib
import time
import matplotlib.pyplot as plt
def conf(y_true,y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(),y_true.max())+1
    w = np.zeros((D,D),dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i],y_true[i]]+=1
    match = linear_assignment(w.max()-w)
    a , b = match
    confuse = np.zeros((D,D),dtype=np.int64)
    for i in range(256):
        confuse[b[i],:]=w[a[i],:]
    return confuse


Label = []
for i in range(256):
    for _ in range(100):
        Label.append(i)

true_label = np.array(Label)


Aggmodel = joblib.load('agg_model/iter_11_Agg_0.96_.pkl')
pre_label = Aggmodel.labels_

confuse = conf(true_label,pre_label)

print(confuse[120:130,120:130])
import matplotlib.pyplot as plt
import itertools
vmax = 100
plt.figure(figsize=(10, 10))

plt.imshow(confuse, cmap=plt.cm.BuGn, vmax=vmax, vmin=0)
plt.title('Test Accuracy 96.23%', fontsize=20)

fontsize = 16

plt.xticks(np.linspace(0,255,6), rotation=45, size=fontsize)
plt.yticks(np.linspace(0,255,6), size=fontsize)

'''
fmt = 'd'
thresh = vmax / 2.
for i, j in itertools.product(range(1, X.shape[0] - 1), range(1, X.shape[1] - 1)):
    plt.text(j, i, format(X[i, j], fmt),
             horizontalalignment="center",
             color="white" if X[i, j] > thresh else "black",
             size=11)
'''

label_size = 18

plt.xlabel('Ground Truth', size=label_size)
plt.ylabel('Predicted Class', size=label_size)


plt.savefig('confuse_img/dynamic_256class.png', bbox_inches='tight')





