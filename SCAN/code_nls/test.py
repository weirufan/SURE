import torch
import torch.nn as nn
import numpy as np
import time
import os
from datetime import datetime
from model import ClusterNet9f,FullyConnectedNet,ClusterNet6c
from tqdm import tqdm
from data_loader import ImagesDataset2, greyscale_make_transforms
import torch.optim as optim
from resnet import ResNet50

from sklearn.metrics import confusion_matrix

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True



def test(Model, Criterion,valid_loader, device):
    time_start = time.time()
    print("Test Start!")
    Model.eval()
    valid_loss = 0.0
    now_num = 0
    acc_sum = 0
    pred_record = []
    label_record = []
    with torch.no_grad():
        for j, data in enumerate(valid_loader, 0):
            x, y= data
            x, y = x.to(device), y.to(device),
            y_pred = Model(x)
            Loss = Criterion(y_pred, y)
            valid_loss += Loss.item() * x.size(0)

            y_pred = y_pred.cpu().detach().numpy()  # batch * 8
            y = y.cpu().detach().numpy()  # batch *1
            acc_sum += np.argmax(y_pred, axis=1) == y
            pred_record.append(np.argmax(y_pred, axis=1))
            label_record.append(y)
            now_num += x.size(0)
            torch.cuda.empty_cache()


    valid_loss = valid_loss / now_num
    valid_acc = acc_sum / now_num
    print("Accuracy:{:.3f}".format(np.sum(pred_record==label_record)/now_num))
    print(valid_loss)
    print(valid_acc)
    ###
    #混淆矩阵
   
    cm = confusion_matrix(pred_record,label_record)
    print(cm)
    ###

    print('Finished Test. Cost time {:.2f} minutes'.format((time.time() - time_start) / 60))
    print('---------------------------------------------------------------------------------------------------')


if __name__ == '__main__':
    import random

    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available. Training on CPU')
    else:
        print('CUDA is available. Training on GPU')

    device = torch.device("cuda:0" if train_on_gpu else "cpu")

    # load_dir = '1122_datad0_test0_2'
    # load_dir = '1122_datad0123_test0123'
    # load_dir = '1129_datad0_test0'
    # load_dir = '1126_datad0235_test0235'
    load_dir = '1129_datad024_test024'

    setup_seed(123)


    input_sz = 64
    output_k = 10

    batch_size = 1

    # dataset_root = ''
    dataset_root ='../../DATASET/DATAD/55'
    tf_train, tf_test = greyscale_make_transforms(input_sz)

    img_test = ImagesDataset2(
        root=dataset_root,
        transform=tf_test,
        train='test'
    )
    dataloader_test = torch.utils.data.DataLoader(
        img_test,
        batch_size=batch_size,
        shuffle=False,
    )



    model = ResNet50(output_k,1)

    model.to(device)
    Criterion = nn.CrossEntropyLoss()

    model_name = os.listdir(os.path.join('checkpoint',load_dir))
    model_name.sort(key = lambda x:int(x[16:-4]))
    model_name = model_name[0]
    print(model_name)
    checkpoint = torch.load(os.path.join('checkpoint',load_dir,model_name),map_location = device)
    model.load_state_dict(checkpoint['state_dict'])


    test(model, Criterion, dataloader_test, device)

