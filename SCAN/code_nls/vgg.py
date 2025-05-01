import torch.nn as nn


class VGGTrunk(nn.Module):
    def __init__(self):
        super(VGGTrunk, self).__init__()

    def _make_layers(self, batch_norm=True):
        layers = []
        in_channels = self.in_channels
        for tup in self.cfg:
            assert len(tup) == 2

            out, dilation = tup
            sz = self.conv_size
            stride = 1
            pad = self.pad  # to avoid shrinking
            if out == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif out == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(
                    in_channels,
                    out,
                    kernel_size=sz,
                    stride=stride,
                    padding=pad,
                    dilation=dilation,
                    bias=False,
                )
                if batch_norm:
                    layers += [
                        conv2d,
                        nn.BatchNorm2d(out, track_running_stats=self.batchnorm_track),
                        nn.ReLU(inplace=True),
                    ]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = out

        return nn.Sequential(*layers)


class VGGTrunk_plus(nn.Module):
    def __init__(self):
        super(VGGTrunk_plus, self).__init__()

    def _make_layers(self):
        in_channels = self.in_channels
        fmodel = None
        if self.input_sz == 64:
            fmodel = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),  # 32 64 64
                nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
                ),  # 64 64 64
                nn.BatchNorm2d(64, track_running_stats=self.batchnorm_track),
                nn.LeakyReLU(inplace=True),
                nn.AvgPool2d(kernel_size=2, stride=2),  # 64 32 32
                nn.Conv2d(
                    in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
                ),  # 128 32 32
                nn.Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),  # 256 32 32
                nn.BatchNorm2d(256, track_running_stats=self.batchnorm_track),
                nn.LeakyReLU(inplace=True),
                nn.AvgPool2d(kernel_size=2, stride=2),  # 256 16 16
                nn.Conv2d(
                    in_channels=256,
                    out_channels=512,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),  # 512 16 16
                nn.Conv2d(
                    in_channels=512,
                    out_channels=1024,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),  # 1024 16 16
                nn.Conv2d(
                    in_channels=1024,
                    out_channels=1024,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),  # 1024 16 16
                nn.BatchNorm2d(1024, track_running_stats=self.batchnorm_track),
                nn.LeakyReLU(inplace=True),
                nn.AvgPool2d(kernel_size=2, stride=2),  # 1024 8 8
            )
        elif self.input_sz == 32:
            fmodel = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),  # 32 32 32
                nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
                ),  # 64 32 32
                nn.Conv2d(
                    in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
                ),  # 128 32 32
                nn.BatchNorm2d(128, track_running_stats=self.batchnorm_track),
                nn.AvgPool2d(kernel_size=2, stride=2),  # 128 16 16
                nn.Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),  # 256 16 16
                nn.Conv2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),  # 256 16 16
                nn.BatchNorm2d(256, track_running_stats=self.batchnorm_track),
                nn.AvgPool2d(kernel_size=2, stride=2),  # 256 8 8
                nn.Conv2d(
                    in_channels=256,
                    out_channels=512,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),  # 512 8 8
                nn.Conv2d(
                    in_channels=512,
                    out_channels=512,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),  # 512 8 8
                nn.Conv2d(
                    in_channels=512,
                    out_channels=1024,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),  # 1024 8 8
                nn.Conv2d(
                    in_channels=1024,
                    out_channels=1024,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),  # 1024 8 8
                nn.BatchNorm2d(1024, track_running_stats=self.batchnorm_track),
                nn.AvgPool2d(kernel_size=2, stride=2),  # 1024 4 4
                nn.LeakyReLU(inplace=True),
            )
        elif self.input_sz == 128:
            fmodel = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=64,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),  # 64 64 64
                nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
                ),  # 64 64 64
                nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
                ),  # 64 64 64
                nn.BatchNorm2d(64, track_running_stats=self.batchnorm_track),
                nn.AvgPool2d(kernel_size=2, stride=2),  # 64 32 32
                nn.Conv2d(
                    in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
                ),  # 128 32 32
                nn.Conv2d(
                    in_channels=128,
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),  # 128 32 32
                nn.Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),  # 256 32 32
                nn.BatchNorm2d(256, track_running_stats=self.batchnorm_track),
                nn.AvgPool2d(kernel_size=2, stride=2),  # 256 16 16
                nn.Conv2d(
                    in_channels=256,
                    out_channels=512,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),  # 512 16 16
                nn.Conv2d(
                    in_channels=512,
                    out_channels=1024,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),  # 1024 16 16
                nn.Conv2d(
                    in_channels=1024,
                    out_channels=1024,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),  # 1024 16 16
                nn.BatchNorm2d(1024, track_running_stats=self.batchnorm_track),
                nn.LeakyReLU(inplace=True),
                nn.AvgPool2d(kernel_size=2, stride=2),  # 1024 8 8
            )
        elif self.input_sz == 256:
            fmodel = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),  # 32 256 256
                nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
                ),  # 64 256 256
                nn.BatchNorm2d(64, track_running_stats=self.batchnorm_track),
                nn.LeakyReLU(inplace=True),
                nn.AvgPool2d(kernel_size=2, stride=2),  # 64 128 128
                nn.Conv2d(
                    in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
                ),  # 128 128 128
                nn.Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),  # 256 64 64
                nn.BatchNorm2d(256, track_running_stats=self.batchnorm_track),
                nn.LeakyReLU(inplace=True),
                nn.AvgPool2d(kernel_size=2, stride=2),  # 256 32 32
                nn.Conv2d(
                    in_channels=256,
                    out_channels=512,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),  # 512 32 32
                nn.Conv2d(
                    in_channels=512,
                    out_channels=1024,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),  # 1024 16 16
                nn.Conv2d(
                    in_channels=1024,
                    out_channels=1024,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),  # 1024 16 16
                nn.BatchNorm2d(1024, track_running_stats=self.batchnorm_track),
                nn.LeakyReLU(inplace=True),
                nn.AvgPool2d(kernel_size=2, stride=2),  # 1024 8 8
            )
        elif self.input_sz == 512:
            fmodel = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),  # 32 512 512
                nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1
                ),  # 64 256 256
                nn.AvgPool2d(kernel_size=2, stride=2),  # 64 128 128
                nn.Conv2d(
                    in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
                ),  # 128 128 128
                nn.AvgPool2d(kernel_size=2, stride=2),  # 128 64 64
                nn.Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),  # 256 64 64
                nn.Conv2d(
                    in_channels=256,
                    out_channels=512,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),  # 512 64 64
                nn.AvgPool2d(kernel_size=2, stride=2),  # 512 32 32
                nn.Conv2d(
                    in_channels=512,
                    out_channels=1024,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),  # 1024 32 32
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
                assert m.track_running_stats == self.batchnorm_track
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
