import torch
import numpy as np
import torch.nn as nn
from vgg import VGGTrunk_plus, VGGNet


class ClusterNetTrunk(VGGTrunk_plus):
    def __init__(self, input_sz):
        super(ClusterNetTrunk, self).__init__()

        self.batchnorm_track = True

        self.conv_size = 3
        self.in_channels = 1
        self.input_sz = input_sz
        self.features = self._make_layers()

    def forward(self, x):
        x = self.features(x)
        bn, nf, h, w = x.size()  # -1 1024 8 8
        x = x.view(bn, nf * h * w)  # -1 1024*8*8
        return x  # -1 1024*8*8


class ClusterNetHead(nn.Module):
    def __init__(self, input_sz, output_k=8):
        super(ClusterNetHead, self).__init__()
        self.input_sz = input_sz
        self.batchnorm_track = True
        self.output_k = output_k
        self.num_sub_heads = 1
        features_sp_size = 8
        num_features = 1024
        if self.input_sz == 24:
            features_sp_size = 3
        elif self.input_sz == 64:
            features_sp_size = 8
        elif self.input_sz == 128:
            features_sp_size = 8
        elif self.input_sz == 256:
            features_sp_size = 8
        elif self.input_sz == 32:
            features_sp_size = 4
        elif self.input_sz == 512:
            features_sp_size = 16

        self.heads = nn.Sequential(
            nn.Linear(
                num_features * features_sp_size * features_sp_size, self.output_k
            ),
            nn.Softmax(dim=1),
        )

    def forward(self, x, kmeans_use_features=False):
        if kmeans_use_features:
            results = x  # duplicates
        else:
            results = self.heads(x)
        return results


class ClusterNet(VGGNet):
    def __init__(self, input_sz, output_k=8):
        super(ClusterNet, self).__init__()
        self.input_sz = input_sz
        self.output_k = output_k
        self.batchnorm_track = True

        self.trunk = ClusterNetTrunk(self.input_sz)
        self.head = ClusterNetHead(self.output_k)

        self._initialize_weights()

    def forward(
        self,
        x,
        kmeans_use_features=False,
        trunk_features=False,
        penultimate_features=False,
    ):
        if penultimate_features:
            print("Not needed/implemented for this arch")
            exit(1)

        x = self.trunk(x)

        if trunk_features:  # for semisup
            return x

        x = self.head(x, kmeans_use_features=kmeans_use_features)  # returns list
        return x


import torch.nn.functional as F


class FullyConnectedNet(nn.Module):
    def __init__(self, num_classes):
        super(FullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(64 * 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    pass
