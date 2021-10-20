import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils

from models.net import FPN as FPN
from models.net import MobileNetV1 as MobileNetV1
from models.net import SSH as SSH


class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    """
    anchor_num实际上=2，所以conv1x1的输出通道是num_anchors * 4 = 8
    """

    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        """
        H'和W'，是变化的，FPN的不同层输出是不同的，Resnet50和MobileNet的初始尺寸不同也会导致不同，这里要清楚
        """

        # M[B,64,H',W']=>[B,8,H',W']
        # R[B,256,H',W']=>[B,8,H',W']
        out = self.conv1x1(x)

        # [B,8,H',W'] => [B,H',W',8]
        # 当调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一模一样，https://zhuanlan.zhihu.com/p/64376950
        out = out.permute(0, 2, 3, 1).contiguous()

        # view函数相当于numpy中resize：
        # 举个某个output的例子：
        # [B,H',W',8] => [B,H'*W'*2,4], 4维变3维了
        # 卧槽，这么reshape也太变态了吧
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):
    def __init__(self, cfg=None, phase='train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace, self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            """
            Resnet50结构：https://s2.ax1x.com/2020/02/23/33V5OH.png
            对应：convX <--->layer{X-1}

            # conv1输出   [H/2,W/2,64]
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # conv2_x输出  [H/4,W/4,256] 
            self.layer1 = self._make_layer(block, 64, layers[0])

            # conv3_x输出  [H/8,W/8,512]         <-----****** 我们要的输出
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,dilate=replace_stride_with_dilation[0])

            # conv4_x输出  [H/16,W/16,1024]      <-----****** 我们要的输出
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,dilate=replace_stride_with_dilation[1])

            # conv5_x输出  [H/32,W/32,2048]      <-----****** 我们要的输出
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,dilate=replace_stride_with_dilation[2])

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            """
            backbone = models.resnet50(pretrained=cfg['pretrain'])

        # mobilenet：    'return_layers' :  {'stage1': 1,   'stage2': 2,      'stage3': 3},
        # resnet50：     'return_layers' :  {'layer2': 1,   'layer3': 2,      'layer4': 3},
        # resnet50对应输出：layer2         :  [H/8,W/8,512] , [H/16,W/16,1024], [H/32,W/32,2048]
        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])

        # resnet50:'in_channel': 256,mobilenet: 32
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,  # resenet50:512; mobilenet:64
            in_channels_stage2 * 4,  # resenet50:1024;mobilenet:128
            in_channels_stage2 * 8,  # resenet50:2048;mobilenet:256
        ]

        out_channels = cfg['out_channel']  # resnet50:256,mobilenet:64

        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self, inputs):

        # resnet50参考图：https://s2.ax1x.com/2020/02/23/33V5OH.png
        # 代码名称：       {'layer2': 1,   'layer3': 2,      'layer4': 3},
        # 图示名称：       conv3_x,        conv4_x,          conv5_x
        # 对应输出：       [H/8,W/8,512] , [H/16,W/16,1024], [H/32,W/32,2048]
        out = self.body(inputs)

        # FPN
        # 这里就用了3个fpn的输出，论文是用了5个
        # 输出：        layer2/conv3_x, layer3/conv4_x,   layer4/conv5_x
        # 输出：        [H/8,W/8,512] , [H/16,W/16,1024], [H/32,W/32,2048]
        # 对应论文里的是：P3,             P4,               P5
        # 未实现论文中的P2
        fpn = self.fpn(out)

        # SSH，不改变宽高，输出通道也都仍然是FPN一样的 R:256,M:64
        # 他们的宽高是不一样的，
        feature1 = self.ssh1(fpn[0])  # M:[H/8,W/8,64],  R[H/8,W/8,256]
        feature2 = self.ssh2(fpn[1])  # M:[H/16,W/16,64],R[H/16,W/16,256]
        feature3 = self.ssh3(fpn[2])  # M:[H/32,W/32,64],R[H/32,W/32,256]

        # 细节：这3个features的**宽高不一样**
        features = [feature1, feature2, feature3]

        # 把3个output（注意，不是一样的宽和高噢），分别，然后concat到一起，太TMD简单粗暴了吧
        # BboxHead也是3个，每个feature对应一个BboxHead
        # BboxHead里面把通道数都变成4！！！内部干的，很暴力
        # （BboxHead）[B,H',W',8] => [B,H'*W'*2,4], 4维变3维了
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)

        # 每个feature的map上的点，都是一个备选anchor？
        # 20.16,
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        # " 卧槽！丫没实现 Dense regression/Mesh Decoder,你看！只有3个输出啊，我去看了一眼，loss，果然也没有看到论文里的L_pixel "
        # 哈哈，很喜欢上面作者的吐槽 ^_^
        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output
