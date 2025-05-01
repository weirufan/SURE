'''
Author: Yuren
Date: 2024-11-02 23:45:21
LastEditors: Yuren
LastEditTime: 2025-04-25 15:12:57
FilePath: /IIC-Integrated/code_super/machine/machine.py
Description: 

Copyright (c) 2025 by Yuren, All Rights Reserved. 
'''
# -*- coding: utf-8 -*-
# @Time : 2024/10/31 18:00
# @Author : Yuren
# @Email : billtxb@outlook.com
# @File : kmeans.py.py

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift
from sklearn.metrics import silhouette_score
import os
from scipy.optimize import linear_sum_assignment as linear_assignment
import time

path = 'data-path'
paths = os.listdir(path)
paths.sort(key=lambda x: int(x[:-4]))
# print(paths)
print(len(paths))
data = []
for p in paths:
    im = Image.open(path + '/' + p)
    print(p)
    im = np.array(im)
    im = im.reshape(-1)
    data.append(im)
data = np.array(data)
print(data.shape)

# compare
clustering_methods = {
    'KMeans': KMeans(n_clusters=8),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
    'Agglomerative': AgglomerativeClustering(n_clusters=8),
    'MeanShift': MeanShift(),
}

results = {}



for method_name, model in clustering_methods.items():
    time1 = time.time()
    labels = model.fit_predict(data)
    elapsed_time = time.time() - time1
    results[method_name] = {
        'labels': labels,
        'time': elapsed_time,
        'silhouette_score': (
            silhouette_score(data, labels) if len(set(labels)) > 1 else -1
        ),
    }
    print(f"{method_name} completed in {elapsed_time:.4f} seconds")


for method_name, result in results.items():
    print(f"\n{method_name} labels (first 100): {result['labels'][:100]}")
    print(f"{method_name} silhouette score: {result['silhouette_score']}")

Label = []
f = open('labels.txt', 'r')
for line in f.readlines():
    Label.append(int(line))
f.close()
true_label = np.array(Label)
print(len(true_label))


def acc(y_true, y_pred):
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
    a, b = ind
    suma = 0
    for i in range(len(a)):
        suma += w[a[i], b[i]]
    return suma * 1.0 / y_pred.size


for method_name, result in results.items():
    print(f"{method_name} accuracy: {acc(true_label, result['labels'])}")
