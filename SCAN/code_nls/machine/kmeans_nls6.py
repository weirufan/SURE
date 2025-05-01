# -*- coding: utf-8 -*-
# @Time : 2024/10/31 18:00
# @Author : Yuren
# @Email : billtxb@outlook.com
# @File : kmeans2.py.py

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift
from sklearn.metrics import silhouette_score,confusion_matrix
import os
from scipy.optimize import linear_sum_assignment as linear_assignment
import time
from tqdm import tqdm

data_name = 'NSL21_sub'
path = '../../../DATASET/'+data_name
paths = os.listdir(path)

txt_path = '../../../DATASET/NSL11_label.txt'

# select_label = [1,2,3,4,5,6]
select_label = [1,3,4,6]
classes = 4

paths.sort(key=lambda x: int(x[:-4]))
# paths = paths[:2000]
'''

1000 class



'''
# temp = paths[:1000]+paths[6000:7000]+paths[12000:13000]+paths[18000:19000]+paths[24000:25000]+paths[30000:31000]
# temp = paths[:1000]+paths[6000:7000]
# paths = temp

print(len(paths))
data = []
for p in tqdm(paths):
    im = Image.open(path+'/'+p)
    im = im.crop((192,192,320,320))
    # print(p)
    im = np.array(im)
    im = im.reshape(-1)
    data.append(im)
data = np.array(data)


Label = []
f = open(txt_path,'r')
ind = 0
select_ind = []
for line in f.readlines():
    line = line.strip('\n')
    line = line.split('\t')
    nl = int(line[0])
    if nl in select_label:
        aid = nl-1
        if aid ==2:
            aid = 1
        elif aid == 3:
            aid = 2
        elif aid ==5:
            aid = 3
        else:
            aid = 0


        Label.append(aid)
        select_ind.append(ind)
    ind +=1
f.close()


data = data[select_ind]

true_label = np.array(Label)

print(len(true_label))

print(data.shape)

# 聚类方法的集合
clustering_methods = {
    'KMeans': KMeans(n_clusters=classes),
    # 'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
    'Agglomerative': AgglomerativeClustering(n_clusters=classes)
}

results = {}


for method_name, model in clustering_methods.items():
    time1 = time.time()
    labels = model.fit_predict(data)
    elapsed_time = time.time() - time1
    results[method_name] = {
        'labels': labels,
        'time': elapsed_time,
        'silhouette_score': silhouette_score(data, labels) if len(set(labels)) > 1 else -1
    }
    print(f"{method_name} completed in {elapsed_time:.4f} seconds")


# 输出每种方法的标签及其轮廓系数
for method_name, result in results.items():
    print(f"\n{method_name} labels (first 100): {result['labels'][:100]}")
    print(f"{method_name} silhouette score: {result['silhouette_score']}")


def optimal_confusion_matrix(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    #res = []
    #a , b = ind
    #for i in range(len(a)):
    #    out_c,gt_c = a[i],b[i]
    #    res.append((out_c,gt_c))
    #return res
    a , b = ind
    # Generate the optimal confusion matrix
    optimal_conf_matrix = w[a[:, None], b]
    return optimal_conf_matrix, sum(w[a[i], b[i]] for i in range(len(a))) / y_pred.size
# 计算每种方法的准确率
# Calculate accuracy and confusion matrix
for method_name, result in results.items():
    conf_matrix, accuracy = optimal_confusion_matrix(true_label, result['labels'])
    print(f"{method_name} accuracy: {accuracy}")
    print(f"{method_name} optimal confusion matrix:\n{conf_matrix}")
