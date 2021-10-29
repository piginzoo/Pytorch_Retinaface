import logging
import time

import cv2
import numpy as np
import torch
import torch.utils.data as data

import utils
from config import CFG
from models.layers.functions.anchor_box import PriorBox, AnchorBox
from utils import get_device
from utils.box_utils import decode, decode_landm
from utils.data.wider_face import WiderFaceValDataset
from utils.nms.py_cpu_nms import py_cpu_nms

logger = logging.getLogger(__name__)


def test(model, image_dir, label_path, batch_size, test_batch_num, anchors, num_workers=1):
    device = utils.get_device()

    dataset = WiderFaceValDataset(image_dir, label_path)
    logger.info("数据集加载完毕：合计 %d 张", len(dataset))
    data_loader = data.DataLoader(dataset,
                                  batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    start = time.time()
    preds = []
    landmarks = []
    gts = []
    for step in range(test_batch_num):
        # 加载一个批次的训练数据
        images, labels = next(data_loader)
        logger.debug("加载了%d张图片, %d个标签s", len(images),len(labels))
        images = images.to(device)
        labels = [anno.to(device) for anno in labels]

        for image, labels_of_image in zip(images,labels):
            bbox_scores, landmark = pred(image, model,anchors)
            logger.debug("预测结果：image : %r", image.shape)
            logger.debug("预测结果：bboxes：%r", landmark.shape)
            logger.debug("预测结果：labels：%d", len(labels_of_image))
            preds.append(bbox_scores)
            landmarks.append(landmark)
            gts.append(labels_of_image)

    logger.info("预测完成，%d 张，耗时： %.2f 分钟", batch_size * test_batch_num, (time.time() - start) / 60)
    return preds,gts


def pred(image, model, anchors, network_config):
    """
    传入图片，得到预测框，已经经过了IOU（CFG.nms_threshold）、概率过滤（CFG.confidence_threshold）。
    预测时，批量处理图片，图片又包含不定数个bbox，所以，bbox的数量是不确定的
    预测时，会同时得到10张图片的bboxes，但是，处理IOU和F1、AP的时候，需要每张图片单独处理。
    预测结果举例: bbox[10, 29126, 4],class [10, 29126, 2], landmark[10, 29126, 10]
    """
    device = get_device()

    image = cv2.resize(image, None, None, fx=network_config.image_size, fy=network_config.image_size,
                       interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = image.shape
    scale = torch.Tensor([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
    image -= (104, 117, 123)
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).unsqueeze(0)
    image = image.to(device)
    scale = scale.to(device)

    # bbox[10, 29126, 4],class [10, 29126, 2], landmark[10, 29126, 10]
    locations, scores, landms = model([image])  # forward pass

def post_process(locations, scores, landms, anchors, network_config):
    device = get_device()

    # # 生成备选anchors：[cx, cy, s_kx, s_ky]
    # anchors = AnchorBox(network_config, image_size=(im_height, im_width)).forward()
    # anchors = anchors.to(device)
    # anchors = anchors.data

    # 根据预测结果，得到调整后的bboxes
    boxes = decode(locations.data.squeeze(0), anchors, network_config['variance'])
    # 按照缩放大小，调整其坐标
    boxes = boxes.cpu().numpy()


    # 计算landmarks
    scores = scores.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), anchors, network_config['variance'])
    landms = landms.cpu().numpy()


    boxes = boxes * scale / network_config.image_size
    scale1 = torch.Tensor([image.shape[3], image.shape[2],
                           image.shape[3], image.shape[2],
                           image.shape[3], image.shape[2],
                           image.shape[3], image.shape[2],
                           image.shape[3], image.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / network_config.image_size

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
    logger.debug("NMS后，剩余人脸框: %d 个", len(bbox_scores))

    # 原代码注释掉了，不过滤NMS后的框了，keep_top_k=750
    # keep top-K faster NMS
    # bbox_scores = bbox_scores[:args.keep_top_k, :]
    # landms = landms[:args.keep_top_k, :]

    return bbox_scores, landms