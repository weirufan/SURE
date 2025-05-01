# -*- codding: utf-8 -*-
'''
@Author : Yuren
@Dare   : 2022/3/11-3:35 下午
'''
import torch.nn as nn

from .net import ClusterNet, ClusterNetTrunk
from .vgg import VGGNet

__all__ = ["ClusterNetTwoHead"]


class ClusterNetTwoHeadHead(nn.Module):
  def __init__(self, config, output_k, semisup=False):
    super(ClusterNetTwoHeadHead, self).__init__()

    self.batchnorm_track = config.batchnorm_track
    num_features = 1024

    self.semisup = semisup

    if config.input_sz == 24:
      features_sp_size = 3
    elif config.input_sz == 64:
      features_sp_size = 8
    elif config.input_sz == 128:
      features_sp_size = 8
    elif config.input_sz == 256:
      features_sp_size = 8
    elif config.input_sz == 32:
      features_sp_size = 4
    elif config.input_sz ==512:
      features_sp_size = 16

    if not semisup:
      self.num_sub_heads = config.num_sub_heads

      self.heads = nn.ModuleList([nn.Sequential(
        nn.Linear(num_features * features_sp_size * features_sp_size, output_k),
        nn.Softmax(dim=1)) for _ in range(self.num_sub_heads)])
    else:
      self.head = nn.Linear(num_features * features_sp_size * features_sp_size, output_k)

  def forward(self, x, kmeans_use_features=False):

    if not self.semisup:
      results = []
      for i in range(self.num_sub_heads):
        if kmeans_use_features:
          results.append(x)  # duplicates
        else:
          results.append(self.heads[i](x))
      return results

    else:
      return self.head(x)


class ClusterNet9fTwoHead(VGGNet):
  def __init__(self, config):
    super(ClusterNet9fTwoHead, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    self.trunk = ClusterNetTrunk(config)

    self.head_A = ClusterNetTwoHeadHead(config, output_k=config.output_k_A)  # train 50

    semisup = (hasattr(config, "semisup") and
               config.semisup)
    print("semisup: %s" % semisup)

    self.head_B = ClusterNetTwoHeadHead(config, output_k=config.output_k_B,semisup=semisup)  # eval 10

    self._initialize_weights()

  def forward(self,
              x,
              head="B",
              kmeans_use_features=False,
              trunk_features=False,
              penultimate_features=False):

    if penultimate_features:
      print("Not needed/implemented for this arch")
      exit(1)

    x = self.trunk(x)

    if trunk_features:  # for semisup
      return x

    # returns list or single
    if head == "A":
      x = self.head_A(x, kmeans_use_features=kmeans_use_features)
    elif head == "B":
      x = self.head_B(x, kmeans_use_features=kmeans_use_features)
    else:
      assert (False)

    return x
