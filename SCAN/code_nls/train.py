import torch
import torch.nn as nn
import numpy as np
import time
import os
from datetime import datetime
from model import ClusterNet9f,FullyConnectedNet,ClusterNet6c
from net.swin_transformer_v2 import  SwinTransformerV2
from resnet import ResNet50
# from tqdm import tqdm
from data_loader import ImagesDataset,greyscale_make_transforms
import torch.optim as optim
def setup_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  torch.backends.cudnn.deterministic = True

def mkdir_savedir(save_name):
    save_path = 'checkpoint/'+save_name
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print("save_path{} create!!!".format(save_path))

def save_model(Model,Optimizer,epoch,save_name,min_loss,valid_loss,train_loss):
    save_Flag = False

    if min_loss > valid_loss:
        if valid_loss<2*train_loss:
            if not save_Flag:
                min_loss = valid_loss
                torch.save({'epoch': epoch + 1,
                            'state_dict': Model.state_dict(),
                            'optimizer': Optimizer.state_dict()
                            }, 'checkpoint/'+save_name + '/best_loss_model_'+str(int(round(min_loss,3)*1000))+'.pth')
            save_Flag = True
    return min_loss,save_Flag

def train(Model, Criterion, Optimizer,Explr, n_epoch, train_loader, valid_loader, device,save_name='AUG'):
    time_start = time.time()
    print("Training Start!")
    mkdir_savedir(save_name)
    min_loss = 1.0


    for epoch in range(n_epoch):
        train_loss = 0.0
        valid_loss = 0.0
        now_num = 0
        Model.train()
        acc_sum = 0
        for i, data in enumerate(train_loader, 0):
            # print(f"test process {i+1}")
            x, y= data
            x, y = x.to(device), y.to(device)
            Optimizer.zero_grad()
            y_pred = Model(x)
            Loss = Criterion(y_pred, y)
            Loss.backward()
            Optimizer.step()
            train_loss += Loss.item() * x.size(0)

            y_pred = y_pred.cpu().detach().numpy()   #batch * 8
            y = y.cpu().detach().numpy()            # batch *1
            # print(np.argmax(y_pred,axis=1))
            # print(y)
            acc_sum += np.sum(np.argmax(y_pred,axis=1)==y)
            now_num += x.size(0)
            #if i %100==0:
            #    print("fc5 weight:", Model.fc5.weight)
            #    print("fc5 bias:", Model.fc5.bias)

            
            torch.cuda.empty_cache()

        train_loss = train_loss / now_num
        train_acc = acc_sum/now_num

        now_num = 0
        acc_sum = 0

        Explr.step()
        Model.eval()

        with torch.no_grad():
            for j, data in enumerate(valid_loader, 0):
                x, y  = data
                x, y = x.to(device), y.to(device),
                y_pred = Model(x)
                Loss = Criterion(y_pred,y)
                valid_loss += Loss.item() * x.size(0)

                y_pred = y_pred.cpu().detach().numpy()  # batch * 8
                y = y.cpu().detach().numpy()  # batch *1
                acc_sum += np.sum(np.argmax(y_pred, axis=1) == y)
                now_num += x.size(0)

                torch.cuda.empty_cache()

        valid_loss = valid_loss / now_num
        valid_acc = acc_sum/now_num

        min_loss,save_Flag = save_model(Model, Optimizer, epoch, save_name,  min_loss, valid_loss,train_loss)

        print('Epoch: {}/{}  Training Loss: {:.6f}  Validation Loss: {:.6f}  Training Acc: {:.6f}  Validation Acc:{:.6f}'
            .format(epoch + 1, n_epoch, train_loss, valid_loss, train_acc, valid_acc))
        print(f"Save Mode:{save_Flag}")
        print("================================================================")

    print('Finished Training. Cost time {:.2f} minutes'.format((time.time() - time_start) / 60))
    print('---------------------------------------------------------------------------------------------------')

if __name__ == '__main__':
    import random
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available. Training on CPU')
    else:
        print('CUDA is available. Training on GPU')

    device = torch.device("cuda:0" if train_on_gpu else "cpu")


    setup_seed(1234)

    input_sz = 64
    output_k =  8


    num_epochs = 100
    batch_size = 128
    Lr = 4e-3
   
    # dataset_root ='../../DATASET/mnist_GG_02345679'
    # dataset_root ='../../DATASET/DATAD/024'
    # dataset_root ='../../DATASET/DATAD/0235'
    dataset_root ='../../DATASET/mnist_zno_0123'
    tf_train,tf_test = greyscale_make_transforms(input_sz)
    img_train = ImagesDataset(
        root=dataset_root,
        transform=tf_train,
        train=True
    )
    img_test = ImagesDataset(
        root=dataset_root,
        transform=tf_test,
        train=False
    )
    dataloader_train = torch.utils.data.DataLoader(
        img_train,
        batch_size=batch_size,
        shuffle=True
    )
    dataloader_test = torch.utils.data.DataLoader(
        img_test,
        batch_size=batch_size,
        shuffle=False,
    )
    '''
    model = SwinTransformerV2(img_size=64, patch_size=4, in_chans=1, num_classes=8,
                  embed_dim=24, depths=[1, 1, 3, 1], num_heads=[3, 6, 12, 24],
                 window_size=4, mlp_ratio=4., qkv_bias=True,
                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                  norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                  use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0])
    '''
    model = FullyConnectedNet(num_classes = output_k)
    # model = ClusterNet9f(input_sz=input_sz, output_k=output_k)
    # model = ResNet50(output_k,1)

    # load_dir = '1128_datad0_test0'
    print(model)
    model.to(device)
    # model_name = os.listdir(os.path.join('checkpoint',load_dir))
    # model_name.sort(key = lambda x:int(x[16:-4]))
    # model_name = model_name[0]
    # print(model_name)
    # checkpoint = torch.load(os.path.join('checkpoint',load_dir,model_name),map_location = device)
    # model.load_state_dict(checkpoint['state_dict'])
    Criterion = nn.CrossEntropyLoss()

    Optimizer = optim.AdamW(model.parameters(), lr=Lr, betas=(0.6, 0.999), eps=1e-08, weight_decay=1e-4,
                            amsgrad=True)
    Explr = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=10, gamma=0.5, last_epoch=-1)


    train(model,Criterion,Optimizer,Explr, num_epochs, dataloader_train, dataloader_test, device,'1201_fcn_bntest')

