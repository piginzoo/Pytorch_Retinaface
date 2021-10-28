import logging
import math

import cv2
import numpy as np
import torch
import torch.utils.data as data

from config import CFG
from data.wider_face import WiderFaceValDataset
from utils import get_device
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms

logger = logging.getLogger(__name__)


def test(image_dir, label_path, batch_size, num_workers=1):
    dataset = WiderFaceValDataset(image_dir, label_path)

    logger.info("数据集加载完毕：合计 %d 张", len(dataset))
    data_loader = iter(data.DataLoader(dataset,
                                       batch_size,
                                       shuffle=True,
                                       num_workers=num_workers))

    steps_of_epoch = math.ceil(len(dataset) / batch_size)
    logger.info("训练集批次：%d 张/批，一个epoch共有 %d 个批次", batch_size, steps_of_epoch)

    # 开始训练！
    for epoch in range(max_epoch):
        logger.info("开始 第 %d 个 epoch", epoch)

        epoch_start = time.time()
        for step in range(steps_of_epoch):
            # if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
            #     torch.save(net.state_dict(), save_folder + cfg['name'] + '_epoch_' + str(epoch) + '.pth')
            # epoch += 1

            # 加载一个批次的训练数据
            images, labels = next(data_loader)
            images = images.to(device)
            labels = [anno.to(device) for anno in labels]
            logger.debug("加载了%d条数据", len(images))


def pred(image, model):
    """
    传入图片，得到预测框，已经经过了IOU（CFG.nms_threshold）、概率过滤（CFG.confidence_threshold）。
    """

    device = get_device()

    # testing scale
    target_size = 1600
    max_size = 2150
    im_shape = image.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    resize = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(resize * im_size_max) > max_size:
        resize = float(max_size) / float(im_size_max)
    if args.origin_size:
        resize = 1

    if resize != 1:
        image = cv2.resize(image, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = image.shape
    scale = torch.Tensor([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
    image -= (104, 117, 123)
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).unsqueeze(0)
    image = image.to(device)
    scale = scale.to(device)

    _t['forward_pass'].tic()
    loc, conf, landms = model(image)  # forward pass
    _t['forward_pass'].toc()
    _t['misc'].tic()

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data

    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([image.shape[3], image.shape[2], image.shape[3], image.shape[2],
                           image.shape[3], image.shape[2], image.shape[3], image.shape[2],
                           image.shape[3], image.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores, confidence_threshold = 0.02
    # 过滤掉人脸概率小于0.02的框
    inds = np.where(scores > CFG.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]
    logger.debug("按照置信度阈值[%.2f]过滤后，剩余人脸框: %d 个", CFG.confidence_threshold, len(scores))

    # keep top-K before NMS
    order = scores.argsort()[::-1]
    # order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

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

    bbox_scores_landmarks = np.concatenate((bbox_scores, landms), axis=1)

    return bbox_scores_landmarks
