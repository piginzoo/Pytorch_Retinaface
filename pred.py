import logging
import time

import cv2
import numpy as np
import torch
import torch.utils.data as data

import utils
from config import CFG
from utils import get_device
from utils.box_utils import decode, decode_landm
from utils.data.wider_face import WiderFaceValDataset
from utils.nms.py_cpu_nms import py_cpu_nms

logger = logging.getLogger(__name__)


def test(model, network_conf, image_dir, label_path, batch_size, test_batch_num, anchors, num_workers=0):
    device = utils.get_device()

    dataset = WiderFaceValDataset(image_dir, label_path)
    logger.info("数据集加载完毕：合计 %d 张", len(dataset))
    data_loader = iter(data.DataLoader(dataset,
                                       batch_size,
                                       shuffle=True,
                                       num_workers=num_workers))
    start = time.time()
    preds = []
    landmarks = []
    gts = []
    for step in range(test_batch_num):
        # 加载一个批次的训练数据,!!! 通过DataLoader加载的变量都会自动变成张量Tensor，靠！
        images, labels = next(data_loader)

        logger.debug("加载了%d张图片, %d个标签s", len(images), len(labels))
        labels = [anno.to(device) for anno in labels]

        for image, labels_of_image in zip(images, labels):
            bbox_scores, landmark = pred(image, model, anchors, network_conf)
            logger.debug("预测结果：image : %r", image.shape)
            logger.debug("预测结果：bboxes：%r", landmark.shape)
            logger.debug("预测结果：labels：%d", len(labels_of_image))
            preds.append(bbox_scores)
            landmarks.append(landmark)
            gts.append(labels_of_image)

    logger.info("预测完成，%d 张，耗时： %.2f 分钟", batch_size * test_batch_num, (time.time() - start) / 60)
    return preds, gts


def pred(image, model, anchors, network_config):
    """
    传入图片，得到预测框，已经经过了IOU（CFG.nms_threshold）、概率过滤（CFG.confidence_threshold）。
    预测时，批量处理图片，图片又包含不定数个bbox，所以，bbox的数量是不确定的
    预测时，会同时得到10张图片的bboxes，但是，处理IOU和F1、AP的时候，需要每张图片单独处理。
    预测结果举例: bbox[10, 29126, 4],class [10, 29126, 2], landmark[10, 29126, 10]
    """
    device = get_device()
    conf_size = network_config['image_size']
    image_original_size = (image.shape[1], image.shape[0])  # W,H, size一般都是(W,H)，按这个顺序来
    logger.debug("图像原shape[%r],size[%r],准备 resize => %r", image.shape, image_original_size, conf_size)

    # resize成网络需要的尺寸
    image = np.array(image)
    image = cv2.resize(image, (conf_size, conf_size))
    logger.debug("图像Resize成[%r]", image.shape)

    # 预处理
    image = image.astype(np.float32)
    image -= (104, 117, 123)  # TODO，为何要做均值化？
    images = np.array([image])  # 增加一个维度，网络要求的
    images = torch.from_numpy(images)
    images = images.to(device)
    images = images.permute(0, 3, 1, 2)  # [1,H,W,C] => [1,C,H,W] ,网络要求的顺序

    # 预测
    # bbox[10, 29126, 4],class [10, 29126, 2], landmark[10, 29126, 10]
    pred_boxes, scores, landms = model(images)  # forward pass

    # 计算缩放scale，未预测完，还原到原图坐标做准备
    size_scale = np.array([conf_size / image_original_size[0], conf_size / image_original_size[1]])

    # 后处理
    pred_boxes_scores, pred_landms = post_process(pred_boxes, scores, landms, anchors, size_scale)

    return pred_boxes_scores, pred_landms


def post_process(locations, scores, landms, anchors, size_scale=None):  # size(W,H)
    """
    一张图片的后处理
    :param locations:
    :param scores:
    :param landms:
    :param anchors:
    :param network_config:
    :return:
    """

    # 根据预测结果，得到调整后的bboxes
    boxes = decode(locations.data.squeeze(0), anchors, CFG.variance)

    # 按照缩放大小，调整其坐标
    boxes = boxes.cpu().numpy()

    # 计算landmarks
    scores = scores.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), anchors, CFG.variance)
    landms = landms.cpu().numpy()

    # 按照图像恢复大小
    if size_scale is not None:
        # boxes: [N,4=x1y1x2y2]
        boxes[:, 0::2] = boxes[:, 0::2] / size_scale[0]  # W
        boxes[:, 1::2] = boxes[:, 1::2] / size_scale[1]  # H
        # landmarks: [N,10] , 奇数/size_scale[0] ，偶数/size_scale[1]
        landms[:, 0::2] = landms[:, 0::2] / size_scale[0]  # W
        landms[:, 1::2] = landms[:, 1::2] / size_scale[1]  # H

    # ignore low scores, confidence_threshold = 0.02
    # 过滤掉人脸概率小于0.02的框
    inds = np.where(scores > CFG.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]
    logger.debug("按照置信度阈值[%.2f]过滤后，剩余人脸框: %d 个", CFG.confidence_threshold, len(scores))

    # keep top-K before NMS
    # order = scores.argsort()[::-1]
    # order = scores.argsort()[::-1][:CFG.top_k]
    # boxes = boxes[order]
    # landms = landms[order]
    # scores = scores[order]

    # 使用NMS算法，过滤重叠框，剩余的框就是概率超过0.02，且概率最大的重叠框
    bbox_scores = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(bbox_scores, CFG.nms_threshold)
    # keep = nms(bbox_scores, args.nms_threshold,force_cpu=args.cpu)
    bbox_scores = bbox_scores[keep, :]
    landms = landms[keep]
    logger.debug("NMS后，剩余人脸框: %d 个, shape:%r", len(bbox_scores), bbox_scores.shape)

    # 原代码注释掉了，不过滤NMS后的框了，keep_top_k=750
    # keep top-K faster NMS
    # bbox_scores = bbox_scores[:args.keep_top_k, :]
    # landms = landms[:args.keep_top_k, :]

    return bbox_scores, landms
