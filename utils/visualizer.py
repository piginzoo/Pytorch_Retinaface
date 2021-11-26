import logging
import os
import warnings
from datetime import datetime

import matplotlib
import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib').disabled = True
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", module="matplotlib")
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

logger = logging.getLogger(__name__)


class TensorboardVisualizer(object):
    """
    参考：https://jishuin.proginn.com/p/763bfbd5447c
    """

    def __init__(self, log_dir):
        self.log_dir = os.path.join(log_dir, datetime.strftime(datetime.now(), '%Y%m%d-%H%M'))

        if not os.path.exists(self.log_dir):    os.makedirs(self.log_dir)

    def text(self, step, value, name):
        # summary_writer = tf.summary.create_file_writer(logdir=self.log_dir)
        # with summary_writer.as_default():
        #     tf.summary.scalar(name, value, step=step)
        # summary_writer.close()
        writer = SummaryWriter(log_dir=self.log_dir)
        writer.add_scalar(name, value, step)
        writer.close()

    def image(self, images, name, step=0):
        """
        :param images: `[h, w, c]`
        """
        # images = np.array(images)
        # if type(images) != np.ndarray:
        #     raise ValueError("图像必须为numpy数组，当前图像为：" + str(type(images)))
        # if len(images.shape) == 3:
        #     images = images[np.newaxis, :]
        # if len(images.shape) != 4:
        #     raise ValueError("图像必须为[B,H,W,C]，当前图像为：" + str(images.shape))

        # summary_writer = tf.summary.create_file_writer(logdir=self.log_dir)
        # with summary_writer.as_default():
        #     if images.shape[1] == 3:  # [B,C,H,W]
        #         images = np.transpose(images, (0, 2, 3, 1))  # [B,C,H,W]=>[B,H,W,C], tf2.x的image通道顺序
        #     r = tf.summary.image(name, images, step)
        # summary_writer.close()
        # if not r: logger.error("保存图片到tensorboard失败：%r", images.shape)
        # return r
        writer = SummaryWriter(log_dir=self.log_dir)
        # if images.shape[1] == 3:  # [B,C,H,W]
        #     images = np.transpose(images, (0, 2, 3, 1))  # [B,C,H,W]=>[B,H,W,C], tf2.x的image通道顺序
        writer.add_images(name, torch.Tensor(images), step)
        writer.close()
