import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(inp, oup, stride=1, leaky=0):
    """
    3x3卷积 + BatchNormal + LeakRelu
    """
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def conv_bn_no_relu(inp, oup, stride):
    """
    3x3卷积 + BatchNormal
    """
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn1X1(inp, oup, stride, leaky=0):
    """
    卷积 + BatchNorm + LeakRelu
    """
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class SSH(nn.Module):
    """
    Context Modelling:
    To enhance the model’s contextual reasoning power for capturing tiny faces [23],
    SSH [36] and PyramidBox [49] applied context modules on feature pyramids
    to enlarge the receptive field from Euclidean grids.
    SSH的名字的由来：M. Najibi, P. Samangouei, R. Chellappa, and L. S. Davis.
                  Ssh: Single stage headless face detector. In ICCV, 2017.
    观察，就是过几个卷积，大小没变，通道数，从256变成了64
    确实是，卷积步长都是1，也没用用maxpool啥的，宽高应该没变
    """

    def __init__(self, in_channel, out_channel):  # out_channel = 256
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        # in:256, out:128
        self.conv3X3 = conv_bn_no_relu(inp=in_channel, oup=out_channel // 2, stride=1)
        # in:256, out:64
        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        # in:64, out:64
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)
        # in:64, out:64
        self.conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
        # in:64, out:64
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

        # final output is 64

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        # 这个就是论文里的把128+64+64 concat到一起，得到256通道的操作，就是那个"C"
        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    """
    就是一个标准的下采样3次，然后再上采样3次，
    按照下采样的尺寸（宽、高）做差值上采样，
    然后两个feature map做相加合并，
    合并后还做了一个3x3的卷积，
    最终，
    输入是：Resnet50:2048,MobileNet:256
    输出是：Resnet50:256,MobileNet:64,840/????
    """

    def __init__(self, in_channels_list, out_channels):
        """
        :param in_channels_list： (resnet50)[512,1024,2048]，(mobilenet)[64,128,256]
        :param out_channels: resnet50:256,mobilenet:64
        """
        super(FPN, self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1

        # in_channels_list(resnet50)[512,1024,2048]
        # 细节很重要，输出都是256channel，这样才可以merge
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky=leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, input):
        # names = list(input.keys())

        # input是一个张量数组，len=[3]
        # 如果是Resnet，数组为：[H/8,W/8,512],[H/16,W/16,1024],[H/32,W/32,2048]
        # 如果是MobileNet，数组为：[H/8,W/8,64],[H/16,W/16,128],[H/32,W/32,256]
        input = list(input.values())

        # output1/2/3，他们除了做卷积，还统一了通道，都是R:256，M:64了
        # 但是宽高没有统一呢
        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        # interpolate实现插值和上采样: https://www.cnblogs.com/wanghui-garcia/p/11399034.html
        # output2[B,C,H,W], size(2,3)=[H,W]
        # 虽然output1/2/3的channel一样的，但是他们的宽高是不一样的，所以要上采样
        # 上采样的方法有：最近邻、线性、双线性...，这里用的是最近邻nearest
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3  # 对应位置相加
        # merge2就是一个3x3卷积+BN+Relu而已，叫merge是幌子，没merge
        # output2 输出为 R:[H/16,W/16,256] M:[H/16,W/16,64]
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        # output2 输出为 R:[H/8,W/8,256] M:[H/8,W/8,64]
        output1 = self.merge1(output1)

        # 输出：
        # R:output1[H/8,W/8,64] , [H/16,W/16,64], [H/32,W/32,64]
        # M:output1[H/8,W/8,256] , [H/16,W/16,256], [H/32,W/32,256]
        # 对应论文里的是：P3, P4, P5，没有实现C6和P2
        # 通道都统一成了mobilenet[64]，resnet50[256]
        out = [output1, output2, output3]
        return out


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky=0.1),  # 3
            conv_dw(8, 16, 1),  # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1),  # 59 + 32 = 91
            conv_dw(128, 128, 1),  # 91 + 32 = 123
            conv_dw(128, 128, 1),  # 123 + 32 = 155
            conv_dw(128, 128, 1),  # 155 + 32 = 187
            conv_dw(128, 128, 1),  # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2),  # 219 +3 2 = 241
            conv_dw(256, 256, 1),  # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x
