import logging
import os
import os.path
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data

import utils

logger = logging.getLogger(__name__)


def load_labels(label_file, image_dir):
    """
    标签格式如下：
    # 0--Parade/0_Parade_marchingband_1_849.jpg
    449 330 122 149 488.906 373.643 0.0 542.089 376.442 0.0 515.031 412.83 0.0 485.174 425.893 0.0 538.357 431.491 0.0 0.82
    # 0--Parade/0_Parade_Parade_0_904.jpg
    361 98 263 339 424.143 251.656 0.0 547.134 232.571 0.0 494.121 325.875 0.0 453.83 368.286 0.0 561.978 342.839 0.0 0.89
    78 221 7 8 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 0.2
    """
    if not os.path.exists(label_file):
        logger.error("标签文件不存在")
        exit()

    f = open(label_file, 'r')
    lines = f.readlines()
    parsed_img_path = False  # 是否刚刚解析了图片路径（而不是标签）

    labels = []
    image_paths = []

    for line in lines:
        line = line.rstrip()
        if line.startswith('#'):  # #号是图片文件名 "# 9--Press_Conference/9_Press_Conference_Press_Conference_9_22.jpg"
            if parsed_img_path:
                logger.warning("正在解析图片路径状态，又是这个状态，忽略上张图片:%s", image_paths.pop())

            path = line[2:]  # 去掉 #和空格，得到文件名
            path = os.path.join(image_dir, path)  # 得到文件全路径
            if not os.path.exists(path):
                raise ValueError("图片不存在：" + path)

            image_paths.append(path)
            labels.append([])  # 出入一个空占位
            parsed_img_path = True
        else:
            # 脸的各种信息：
            # 长度是20: xywh:4 , landmark:10(5x2), 分割的'0.0'：5，置信度：1
            # 实际需要保存的也就是4+10=14个就行
            # 427 46 141 194 469.688 118.125 0.0 534.281 127.875 0.0 498.938 164.438 0.0 469.688 186.375 0.0 523.312 191.25 0.0 0.9
            line = line.split(' ')
            label = [float(x) for x in line]
            labels[-1].append(label)
            parsed_img_path = False

    logger.info("加载原始标注[%d]条，图像路径[%d]条", len(labels), len(image_paths))
    assert len(labels) == len(image_paths), str(len(labels)) + "/" + str(len(image_paths))
    return labels, image_paths


class WiderFaceTrainDataset(data.Dataset):
    """
    加载RetinaFace专门重新标注过的WiderFace数据集：

    1、人脸标注

    格式：box（x1, y1, w, h），紧接着跟着5个landmark，彼此用0.0分割，最后是一个置信度

    例子：
    ```
        # 0--Parade/0_Parade_marchingband_1_849.jpg
        449 330 122 149 488.906 373.643 0.0 542.089 376.442 0.0 515.031 412.83 0.0 485.174 425.893 0.0 538.357 431.491 0.0 0.82
        ~~~~人脸bbox~~~~ ~~~lanmark1~~~     ~~~lanmark2~~~      ~~~lanmark3~~~      ~~~lanmark4~~~      ~~~lanmark5~~~     ~置信度

        449 330 122 149 表示box（x1, y1, w, h）
        接着是5个关键点信息，分别用0.0隔开 或者1.0分开
        488.906 373.643 0.0
        542.089 376.442 0.0
        515.031 412.83 0.0
        485.174 425.893 0.0
        538.357 431.491 0.0
    ```

    """

    def __init__(self, train_image_dir, train_label, preproc=None):
        self.preproc = preproc
        self.face_annonation, self.imgs_path = load_labels(train_label, train_image_dir)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        """
        是的，证明了我上面的推测，两个数组self.face_annonation，self.imgs_path，分别存在人脸和图片路径
        """

        if not os.path.exists(self.imgs_path[index]):
            logger.warning("图片不存在：%r", self.imgs_path[index])
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.face_annonation[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            logger.warning("这张图片不包含任何人脸")
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))  # 长度是15个
            # bbox ： 0 - 3，4个
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks： 4 - 13 10个
            annotation[0, 4] = label[4]  # l0_x
            annotation[0, 5] = label[5]  # l0_y
            annotation[0, 6] = label[7]  # l1_x
            annotation[0, 7] = label[8]  # l1_y
            annotation[0, 8] = label[10]  # l2_x
            annotation[0, 9] = label[11]  # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y

            # 最后一个位，是置信度
            if (annotation[0, 4] < 0):
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        faces_info = np.array(annotations)
        if self.preproc is not None:
            img, faces_info = self.preproc(img, faces_info)

        return img, faces_info


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    WiderFaceTrainDataset返回的是1张图，和，个数不确定的bboxes，
    现在需要把他们合并到一起，24张图是一批，
    图片比较容易concat到一起，
    但是bboxes们数量不定，

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    # logger.debug("batch size:%d", len(batch))
    for i, sample in enumerate(batch):
        img, annonation = sample
        imgs.append(img)
        # import sys
        # logger.debug("图像#%d内存大小：%r",i+1, sys.getsizeof(img))
        annos = torch.from_numpy(annonation).float()
        targets.append(annos)

    # logger.debug("处理完1：Images:%r, Labels:%r", len(imgs), len(targets))

    if len(imgs) == 0:
        logger.warning("batch中的图片为0，batch大小为：%d", len(batch))

    imgs = np.array(imgs)
    # logger.debug("处理完2：Images:%r, Labels:%r,大小：%r", len(imgs), len(targets),sys.getsizeof(imgs))
    imgs = torch.from_numpy(imgs).float()
    # logger.debug("返回结果：Images:%r, Labels:%r,大小:%r", imgs.shape,len(targets),sys.getsizeof(imgs))
    return imgs, targets


class WiderFaceValDataset(list):
    """
    加载RetinaFace验证(Val)用WiderFace数据集：

    格式：box（x1, y1, w, h），没有训练集中的5个landmark

    例子：
    ```
        # 0--Parade/0_Parade_marchingband_1_849.jpg
        449 330 122 149
        488 906 373 643
        ...
    ```
    """

    def __init__(self, image_dir, label_path):
        self.labels_images = list(zip(load_labels(label_path, image_dir)))
        logger.debug("创建数据集[%s],图片/标注[%d]条", label_path, len(self.labels_images))

    def __len__(self):
        return len(self.labels_images)

    def shuffle(self):
        random.shuffle(self.labels_images)

    def __getitem__(self, index):
        """
        是的，证明了我上面的推测，两个数组self.face_annonation，self.imgs_path，分别存在人脸和图片路径
        """

        labels, image_path = self.labels_images[index]

        img = cv2.imread(image_path)
        height, width, _ = img.shape

        annotations = np.zeros((1, 4))
        if len(labels) == 0:
            logger.warning("图片[%s]的标注为空",image_path)
            return annotations

        # logger.debug("labels:%r",labels)
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 4))  # 长度是4个
            # logger.debug("annotation[0, 0] = label[0] :%r/%r",annotation,label)
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2
            annotations = np.append(annotations, annotation, axis=0)
        faces_info = np.array(annotations)
        return img, faces_info


# python -m utils.data.wider_face
if __name__ == '__main__':
    utils.init_log()
    labels, image_paths = load_labels(label_file="data/label.retina/train/label.txt",
                                      image_dir="data/images/train")
    assert len(labels) == len(image_paths), str(len(labels)) + "/" + str(len(image_paths))
    for label, image_path in zip(labels, image_paths):
        print(image_path)
        if type(label) == list:
            for l in label:
                print("\t", l)

    labels, image_paths = load_labels(label_file="data/label.retina/val/label.txt",
                                      image_dir="data/images/val")
    assert len(labels) == len(image_paths), str(len(labels)) + "/" + str(len(image_paths))
