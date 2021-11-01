from itertools import product as product
from math import ceil

import torch


class AnchorBox(object):
    """
    生成anchors
    """

    def __init__(self, cfg, image_size=None, phase='train'):
        super(AnchorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']  # 'min_sizes': [[16, 32], [64, 128], [256, 512]],
        self.steps = cfg['steps']  # 'steps': [8, 16, 32] <-- 原图缩放的大小
        self.image_size = image_size

        # [[105,105],[53,53],[27,27]]
        self.feature_maps = [[ceil(self.image_size[0] / step),
                              ceil(self.image_size[1] / step)]
                             for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, feature_map_size in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            # 展开feature map中的每个点，[i,j]就是像素坐标
            for i, j in product(range(feature_map_size[0]),
                                range(feature_map_size[1])):  # product(A,B)函数,返回A和B中的元素组成的笛卡尔积的元组

                # 'min_sizes': [[16, 32], [64, 128], [256, 512]]
                # 每个尺度尝试两个大小：以第一个举例，16像素和32像素
                for min_size in min_sizes:
                    # anchor的宽和高
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    # anchor的中心坐标，这个写法不太懂？？？回头调试一下理解
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    # 感觉就是完成了所有的anchor准备
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        # 我给去掉了，默认是false，干嘛要处理？！
        # if self.clip:
        #     output.clamp_(max=1, min=0)
        return output
