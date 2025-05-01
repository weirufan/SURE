'''
yuren
2022.04.04
'''
import numpy as np
from PIL import Image
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
import os
from scipy.optimize import linear_sum_assignment as linear_assignment
import joblib
import time
import matplotlib.pyplot as plt
path = 'OUT'
paths = os.listdir(path)
paths.sort(key=lambda x: int(x[:-4]))
print("image numbers:{}".format(len(paths)))
data = []
time1 = time.time()
print('Start load image')
for p in paths:
    im = Image.open(path+'/'+p)
    im = np.array(im)
    #im = im[64:192,64:192]
    #print(im.shape)
    #exit(0)
    im = im.reshape(-1)
    data.append(im)
data = np.array(data)
print('image load OVER cost time{}'.format(time.time()-time1))
print('image shape:{}'.format(data.shape))
time2 = time.time()
print('AgglomerativeClustering Start!!!')
#spectral = SpectralClustering(n_clusters=10,assign_labels='discretize',random_state=0)
spectral = AgglomerativeClustering(n_clusters=10)
spectral.fit(data)
print('AgglomerativeClustering over cost time{}'.format(time.time()-time2))
pre_label = spectral.labels_

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
#print(len(true_label))

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
accuracy = acc(true_label,pre_label)
print('the acc:{}'.format(accuracy))
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
    for i in range(10):
        confuse[b[i],:]=w[a[i],:]
    return confuse,match
confuse ,match = conf(true_label,pre_label)
print(confuse)
print('the pred match true : {}'.format(match))
#print(w)
modelpath = 'model/AgglomerativeClustering_'+str(accuracy)+'_.pkl'
joblib.dump(spectral,modelpath)

import itertools
# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Confusion_matrix_reflection3.png')

classes = ['0','1','2','3','4','5','6','7','8','9']
plot_confusion_matrix(confuse, classes=classes, normalize=False, title='Confusion Matrix')
