import torch.nn as nn

from .vgg import VGGTrunk_plus, VGGNet


__all__ = ["ClusterNet"]


class ClusterNetTrunk(VGGTrunk_plus):
  def __init__(self, config):
    super(ClusterNetTrunk, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    self.conv_size = 3
    self.in_channels = config.in_channels if hasattr(config, 'in_channels') \
      else 3
    self.input_sz = config.input_sz

    self.features = self._make_layers()

  def forward(self, x):
    x = self.features(x)
    bn, nf, h, w = x.size()  # -1 1024 8 8
    x = x.view(bn, nf * h * w)  # -1 1024*8*8
    return x   # -1 1024*8*8


class ClusterNetHead(nn.Module):
  def __init__(self, config):
    super(ClusterNetHead, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    self.num_sub_heads = config.num_sub_heads

    num_features = 1024

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

    self.heads = nn.ModuleList([nn.Sequential(
      nn.Linear(num_features * features_sp_size * features_sp_size,config.output_k),
      nn.Softmax(dim=1)) for _ in range(self.num_sub_heads)])

  def forward(self, x, kmeans_use_features=False):
    results = []
    for i in range(self.num_sub_heads):
      if kmeans_use_features:
        results.append(x)  # duplicates
      else:
        results.append(self.heads[i](x))
    return results


class ClusterNet(VGGNet):

  def __init__(self, config):
    super(ClusterNet, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    self.trunk = ClusterNetTrunk(config)
    self.head = ClusterNetHead(config)

    self._initialize_weights()

  def forward(self, x, kmeans_use_features=False, trunk_features=False,
              penultimate_features=False):
    if penultimate_features:
      print("Not needed/implemented for this arch")
      exit(1)

    x = self.trunk(x)

    if trunk_features:  # for semisup
      return x

    x = self.head(x, kmeans_use_features=kmeans_use_features)  # returns list
    return x
