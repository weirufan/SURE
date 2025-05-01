import torch.nn as nn


class VGGTrunk(nn.Module):
  def __init__(self):
    super(VGGTrunk, self).__init__()

  def _make_layers(self, batch_norm=True):
    layers = []
    in_channels = self.in_channels
    for tup in self.cfg:
      assert (len(tup) == 2)

      out, dilation = tup
      sz = self.conv_size
      stride = 1
      pad = self.pad  # to avoid shrinking

      if out == 'M':
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
      elif out == 'A':
        layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
      else:
        conv2d = nn.Conv2d(in_channels, out, kernel_size=sz,
                           stride=stride, padding=pad,
                           dilation=dilation, bias=False)
        if batch_norm:
          layers += [conv2d, nn.BatchNorm2d(out,track_running_stats=self.batchnorm_track),
                     nn.ReLU(inplace=True)]
        else:
          layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = out

    return nn.Sequential(*layers)

class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
    super(Bottleneck, self).__init__()

    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    self.batch_norm1 = nn.BatchNorm2d(out_channels)

    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
    self.batch_norm2 = nn.BatchNorm2d(out_channels)

    self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
    self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

    self.i_downsample = i_downsample
    self.stride = stride
    self.relu = nn.ReLU()

  def forward(self, x):
    identity = x.clone()
    x = self.relu(self.batch_norm1(self.conv1(x)))

    x = self.relu(self.batch_norm2(self.conv2(x)))

    x = self.conv3(x)
    x = self.batch_norm3(x)

    # downsample if needed
    if self.i_downsample is not None:
      identity = self.i_downsample(identity)
    # add identity
    x += identity
    x = self.relu(x)

    return x


class Block(nn.Module):
  expansion = 1

  def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
    super(Block, self).__init__()

    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
    self.batch_norm1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
    self.batch_norm2 = nn.BatchNorm2d(out_channels)

    self.i_downsample = i_downsample
    self.stride = stride
    self.relu = nn.ReLU()

  def forward(self, x):
    identity = x.clone()

    x = self.relu(self.batch_norm2(self.conv1(x)))
    x = self.batch_norm2(self.conv2(x))

    if self.i_downsample is not None:
      identity = self.i_downsample(identity)
    print(x.shape)
    print(identity.shape)
    x += identity
    x = self.relu(x)
    return x

class ResNet(nn.Module):
  def __init__(self, ResBlock, layer_list, num_channels=3):
    super(ResNet, self).__init__()
    self.in_channels = 64

    self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.batch_norm1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU()
    self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
    self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
    self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
    self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


  def forward(self, x):
    x = self.relu(self.batch_norm1(self.conv1(x)))
    x = self.max_pool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)

    return x

  def _make_layer(self, ResBlock, blocks, planes, stride=1):
    ii_downsample = None
    layers = []

    if stride != 1 or self.in_channels != planes * ResBlock.expansion:
      ii_downsample = nn.Sequential(
        nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
        nn.BatchNorm2d(planes * ResBlock.expansion)
      )

    layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
    self.in_channels = planes * ResBlock.expansion

    for i in range(blocks - 1):
      layers.append(ResBlock(self.in_channels, planes))

    return nn.Sequential(*layers)

def ResNet50(channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], channels)

class VGGTrunk_plus(nn.Module):
  def __init__(self):
    super(VGGTrunk_plus, self).__init__()

  def _make_layers(self):
    in_channels = self.in_channels
    if self.input_sz == 64:
      fmodel = ResNet50(in_channels)
      # fmodel = nn.Sequential(
      #    nn.Conv2d(in_channels=in_channels,out_channels=32,kernel_size=3,stride=1,padding=1), #32 64 64
      #    nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),      #64 64 64
      #    nn.BatchNorm2d(64, track_running_stats=self.batchnorm_track),
      #    nn.LeakyReLU(inplace=True),
      #    nn.AvgPool2d(kernel_size=2,stride=2),      # 64 32 32
      #    nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),   # 128 32 32
      #    nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),   # 256 32 32
      #    nn.BatchNorm2d(256, track_running_stats=self.batchnorm_track),
      #    nn.LeakyReLU(inplace=True),
      #    nn.AvgPool2d(kernel_size=2, stride=2),  # 256 16 16
      #    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # 512 16 16
      #    nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),  # 1024 16 16
      #    nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),  # 1024 16 16
      #    nn.BatchNorm2d(1024, track_running_stats=self.batchnorm_track),
      #    nn.LeakyReLU(inplace=True),
      #    nn.AvgPool2d(kernel_size=2, stride=2),  # 1024 8 8
      #  )
      '''
      fmodel = nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=32,kernel_size=3,stride=1,padding=1), #32 64 64
        nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),      #64 64 64
        nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),      #64 64 64
        nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),      #64 64 64
        nn.BatchNorm2d(64, track_running_stats=self.batchnorm_track),
        nn.LeakyReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2,stride=2),      # 64 32 32
        nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),   # 128 32 32
        nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),   # 128 32 32
        nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),   # 256 32 32
        nn.BatchNorm2d(256, track_running_stats=self.batchnorm_track),
        nn.LeakyReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2),  # 256 16 16
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # 512 16 16
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # 512 16 16
        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),  # 1024 16 16
        nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),  # 1024 16 16
        nn.BatchNorm2d(1024, track_running_stats=self.batchnorm_track),
        nn.LeakyReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2),  # 1024 8 8
      )
      fmodel = nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=32,kernel_size=3,stride=1,padding=1), #32 64 64
        nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),      #64 64 64
        nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),      #64 64 64
        nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),      #64 64 64
        nn.BatchNorm2d(64, track_running_stats=self.batchnorm_track),
        nn.AvgPool2d(kernel_size=2,stride=2),      # 64 32 32
        nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),   # 128 32 32
        nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),   # 256 32 32
        nn.BatchNorm2d(256, track_running_stats=self.batchnorm_track),
        nn.AvgPool2d(kernel_size=2, stride=2),  # 256 16 16
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # 512 16 16
        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),  # 1024 16 16
        nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),  # 1024 8 8
        nn.BatchNorm2d(1024, track_running_stats=self.batchnorm_track),
        nn.LeakyReLU(inplace=True),
      )
      '''
      # beta 1   nn.Linear(1024*8*8,10) 单分类头，，根据经验改为mlp可能会达到更好的效果。 32.2 效果不佳
      # beta 2
    elif self.input_sz == 32:
      fmodel = nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=32,kernel_size=3,stride=1,padding=1), # 32 32 32
        nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1), # 64 32 32
        nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),  # 128 32 32
        nn.BatchNorm2d(128, track_running_stats=self.batchnorm_track),
        nn.AvgPool2d(kernel_size=2,stride=2), #128 16 16
        nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),   #256 16 16
        nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),  # 256 16 16
        nn.BatchNorm2d(256,track_running_stats=self.batchnorm_track),
        nn.AvgPool2d(kernel_size=2,stride=2),  # 256 8 8
        nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1), # 512 8 8
        nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1), # 512 8 8
        nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,stride=1,padding=1), # 1024 8 8
        nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=1,padding=1), # 1024 8 8
        nn.BatchNorm2d(1024, track_running_stats=self.batchnorm_track),
        nn.AvgPool2d(kernel_size=2, stride=2),  # 1024 4 4
        nn.LeakyReLU(inplace=True),

      )
    elif self.input_sz == 128:
      fmodel = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=2, padding=1),  # 64 64 64
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # 64 64 64
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # 64 64 64
        nn.BatchNorm2d(64, track_running_stats=self.batchnorm_track),
        nn.AvgPool2d(kernel_size=2, stride=2),  # 64 32 32
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # 128 32 32
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),  # 128 32 32
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # 256 32 32
        nn.BatchNorm2d(256, track_running_stats=self.batchnorm_track),
        nn.AvgPool2d(kernel_size=2, stride=2),  # 256 16 16
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # 512 16 16
        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),  # 1024 16 16
        nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),  # 1024 16 16
        nn.BatchNorm2d(1024, track_running_stats=self.batchnorm_track),
        nn.LeakyReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2),  # 1024 8 8
      )
    elif self.input_sz == 256:
      fmodel = nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=32,kernel_size=3,stride=1,padding=1), #32 256 256
        nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),      #64 256 256
        nn.BatchNorm2d(64, track_running_stats=self.batchnorm_track),
        nn.LeakyReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2,stride=2),      # 64 128 128
        nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),   # 128 128 128
        nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1),   # 256 64 64
        nn.BatchNorm2d(256, track_running_stats=self.batchnorm_track),
        nn.LeakyReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2),  # 256 32 32
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # 512 32 32
        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1),  # 1024 16 16
        nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),  # 1024 16 16
        nn.BatchNorm2d(1024, track_running_stats=self.batchnorm_track),
        nn.LeakyReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2),  # 1024 8 8
      )
    elif self.input_sz == 512:
      fmodel = nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=32,kernel_size=3,stride=1,padding=1), #32 512 512
        nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=2,padding=1),      #64 256 256
        nn.AvgPool2d(kernel_size=2,stride=2),      # 64 128 128
        nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),   # 128 128 128
        nn.AvgPool2d(kernel_size=2, stride=2),  # 128 64 64
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # 256 64 64
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # 512 64 64
        nn.AvgPool2d(kernel_size=2, stride=2),  # 512 32 32
        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),  # 1024 32 32
        nn.AvgPool2d(kernel_size=2, stride=2),  # 1024 16 16
      )
    return fmodel


class VGGNet(nn.Module):
  def __init__(self):
    super(VGGNet, self).__init__()

  def _initialize_weights(self, mode='fan_in'):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity='relu')
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        assert (m.track_running_stats == self.batchnorm_track)
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()
