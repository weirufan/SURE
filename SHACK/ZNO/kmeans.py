'''
yuren
2022.04.04
'''
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import os
from scipy.optimize import linear_sum_assignment as linear_assignment
import time
path = 'OUT'
paths = os.listdir(path)
paths.sort(key=lambda x: int(x[:-4]))
#print(paths)
print(len(paths))
data = []
for p in paths:
    im = Image.open(path+'/'+p)
    print(p)
    im = np.array(im)
    im = im.reshape(-1)
    data.append(im)
data = np.array(data)
print(data.shape)

time1 = time.time()
kmeans = KMeans(n_clusters=10)
kmeans.fit(data)
print(time.time()-time1)
print('over')
print(kmeans.labels_[:100])
pre_label = kmeans.labels_

Label = []
f = open('Label_mnist.txt','r')
for line in f.readlines():
    line = line.strip('\n')
    line = line.split('\t')
    for i in range(10):
        if int(line[i])==1:
            Label.append(i)
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
    #res = []
    #a , b = ind
    #for i in range(len(a)):
    #    out_c,gt_c = a[i],b[i]
    #    res.append((out_c,gt_c))
    #return res
    a , b = ind
    suma = 0
    for i in range(len(a)):
        suma += w[a[i],b[i]]
    return suma*1.0/y_pred.size
print(acc(true_label,pre_label))
