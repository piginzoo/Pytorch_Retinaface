from __future__ import print_function

import argparse
import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data

import config
import eval
import pred
from models.layers.functions.anchor_box import AnchorBox
from models.layers.modules import MultiBoxLoss
from models.retinaface import RetinaFace
from utils import init_log, get_device, save_model, image_utils, box_utils
from utils.data import WiderFaceTrainDataset, detection_collate, preproc
from utils.early_stop import EarlyStop
from utils.visualizer import TensorboardVisualizer

init_log()

logger = logging.getLogger(__name__)


def parse_argumens():
    parser = argparse.ArgumentParser(description='Retinaface Training')
    parser.add_argument('--name', default='retinaface')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--network', default='resnet'),
    parser.add_argument('--train_label', default='./data/label.retina/train/label.txt',
                        help='Training dataset directory')
    parser.add_argument('--val_label', default='./data/label.retina/val/label.txt',
                        help='Training dataset directory')
    parser.add_argument('--train_dir', default='./data/images/train/',
                        help='Training dataset directory')
    parser.add_argument('--val_dir', default='./data/images/val/',
                        help='validation dataset directory')
    parser.add_argument('--save_folder', default='./model/', help='Location to save checkpoint models')

    args = parser.parse_args()
    logger.debug("参数：%r", args)
    return args


def train(args):
    # 参数准备
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    device = get_device()
    rgb_mean = (104, 117, 123)  # bgr order
    num_classes = 2
    train_dir = args.train_dir
    train_label = args.train_label
    network_conf = config.network_conf(args.network)
    max_epoch = config.CFG.max_epochs
    num_workers = config.CFG.num_workers
    batch_size = network_conf['batch_size']
    print_steps = config.CFG.print_steps

    # 网络创建
    net = RetinaFace(cfg=network_conf)
    net = net.to(device)

    # TODO 暂不考虑再训练
    # if args.resume_net is not None:
    #     state_dict = torch.load(args.resume_net)
    #     # create new OrderedDict that does not contain `module.`
    #     from collections import OrderedDict
    #
    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         head = k[:7]
    #         if head == 'module.':
    #             name = k[7:]  # remove `module.`
    #         else:
    #             name = k
    #         new_state_dict[name] = v
    #     net.load_state_dict(new_state_dict)

    cudnn.benchmark = True

    multi_box_loss = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)
    multi_box_loss = multi_box_loss.to(device)

    img_size = network_conf['image_size']
    anchor_boxes = AnchorBox(network_conf, image_size=(img_size, img_size))
    with torch.no_grad():
        anchors = anchor_boxes.forward()
        anchors = anchors.to(device)

    visualizer = TensorboardVisualizer(config.CFG.tboard_dir)
    early_stopper = EarlyStop(max_retry=config.CFG.early_stop)

    net.train()
    logger.debug('开始加载数据集: 图片目录：%s，标签文件：%s', train_dir, train_label)

    if args.debug:
        logger.debug(">>>>>>>>>>>>>>>>>>>>>>>>> 调试模式！！！")
        batch_size = 1
        max_epoch = 1
        num_workers = 0
        print_steps = 1

    dataset = WiderFaceTrainDataset(train_dir, train_label, preproc(img_size, rgb_mean))
    logger.info("数据集加载完毕：合计 %d 张", len(dataset))
    data_loader = iter(data.DataLoader(dataset,
                                       batch_size,
                                       shuffle=True,
                                       num_workers=num_workers,
                                       collate_fn=detection_collate))
    steps_of_epoch = math.ceil(len(dataset) / batch_size)

    if args.debug:
        steps_of_epoch = 1

    logger.info("训练集批次：%d 张/批，一个epoch共有 %d 个批次", batch_size, steps_of_epoch)

    # if args.resume_epoch > 0:
    #     total_steps = args.resume_epoch * steps_of_epoch
    # else:
    #     total_steps = 0
    total_steps = 0

    logger.debug('开始训练...')

    # 创建优化器
    optimizer = torch.optim.Adam([{'params': net.parameters(), 'lr': 0.01}])

    # 开始训练！
    min_loss = latest_loss = sys.maxsize

    for epoch in range(1, max_epoch + 1):
        logger.info("================ 开始 第 %d 个 epoch ================ ", epoch)

        epoch_start = time.time()
        for step in range(steps_of_epoch):

            logger.info("------------ 开始 第 %d 步 of epoch %d ------------", steps_of_epoch, epoch)

            # 加载一个批次的训练数据
            images, labels = next(data_loader)
            images = images.to(device)
            labels = [anno.to(device) for anno in labels]
            logger.debug("加载了%d条数据", len(images))

            # 前向运算
            # out = (bbox[N,4,2], class[N,2], landmark[N,5,2])
            # 运行日志：完成前向计算: bbox[1, 29126, 4],class [1, 29126, 2], landmark[1, 29126, 10]
            # 29126怎么来的？很关键！
            net_out = net(images)
            logger.debug("完成前向计算:bbox[%r],class[%r],landmark[%s]", net_out[0].shape, net_out[1].shape, net_out[2].shape)

            # 反向传播
            optimizer.zero_grad()
            loss_l, loss_c, loss_landm = multi_box_loss(net_out, anchors, labels)
            loss = config.CFG.location_weight * loss_l + loss_c + loss_landm
            loss.backward()
            logger.debug("完成反向传播计算")
            optimizer.step()

            total_steps += 1

            # 每隔N个batch，就算一下这个批次的正确率
            if total_steps % print_steps == 0:
                logger.debug("Epoch/Step: %r/%r, 总Step:%r, loss[bbox/class/landmark]: %.4f,%.4f,%.4f", epoch, step,
                             total_steps, loss_l.item(), loss_c.item(), loss_landm.item())
                preds_of_images, scores_of_images, landms_of_images = net_out
                # 需要做一个softmax分类
                scores_of_images = F.softmax(scores_of_images)

                # 逐张图片处理
                for image, pred_boxes, scores, pred_landms, gts in \
                        zip(images, preds_of_images, scores_of_images, landms_of_images, labels):
                    # 预测后处理
                    pred_boxes_scores, pred_landms = pred.post_process(pred_boxes, scores, pred_landms, anchors)
                    # 记录调试信息到tensorboard
                    train_check(visualizer, image, pred_boxes_scores, pred_landms, gts, loss, epoch, total_steps)

                # 从第3个epoch，才开始记录最小loss的模型，且F1设置为0
                if epoch > 2 and latest_loss < min_loss:
                    logger.info("Step[%d] loss[%.4f] 比之前 loss[%.4f] 都低，保存模型", epoch, latest_loss, min_loss)
                    min_loss = latest_loss
                    save_model(net, args.save_folder, epoch, total_steps, latest_loss, 0)

        logger.info("Epoch [%d] 结束，耗时 %.2f 分", epoch, (time.time() - epoch_start) / 60)

        # 做F1的计算，并可视化图片
        validate_start = time.time()
        # 图片目录用的是train_dir，因为图片目录，val和train是共享的
        precision, recall, f1 = validate(net, args.val_dir, args.val_label, network_conf, config.CFG, anchors,
                                         args.debug, num_workers)
        visualizer.text(total_steps, precision, name='Precision')
        visualizer.text(total_steps, recall, name='Recall')
        visualizer.text(total_steps, f1, name='F1')
        logger.info("验证结束，Epoch [%d] ，耗时 %.2f 秒", epoch, time.time() - validate_start)

        # early_stopper可以帮助存基于acc的best模型
        if early_stopper.decide(f1, save_model, net, epoch + 1, total_steps, latest_loss, f1):
            logger.info("早停导致退出：epoch[%d] f1[%.4f]", epoch + 1, f1)
            break
    logger.info("训练结束", epoch)


def validate(model, image_dir, label_path, network_conf, CFG, anchors, is_debug, num_workers):
    batch_size = CFG.val_batch_size
    batch_num = CFG.val_batch_num
    if is_debug:
        batch_size = 1
        batch_num = 1

    logger.debug("............ 开始做validation ............", )

    # 加载一大批图片，算出他们的人脸框
    all_preds, all_gts = pred.test(model, network_conf, image_dir, label_path, batch_size, batch_num, anchors,
                                   num_workers)

    pred_count = gt_count = TP_count = 0

    # 处理每张图片
    for preds, gts in zip(all_preds, all_gts):

        preds = np.array(preds)
        gts = np.array(gts)

        logger.debug("预测出 %d 个框",len(preds))

        # preds[N,5] : [x1,y1,x2,y2,score]
        # 只保留>0.5置信度的框
        preds = preds[preds[:, 4] > 0.5]

        logger.debug("过滤剩余 %d 个框(score>0.5)",len(preds))

        iou_matrx = eval.calc_iou_matrix(preds, gts, 0.5)
        p, r, f, TP = eval.calc_precision_recall(iou_matrx)

        TP_count += TP
        pred_count += iou_matrx.shape[0]
        gt_count += iou_matrx.shape[1]
        logger.debug("图片精确率[%.3f],召回率[%.3f],F1[%.3f],预测[%d]框,GT[%d]框,正确[%d]框",
                     p, r, f, iou_matrx.shape[0], iou_matrx.shape[1], TP)

    precision = TP_count / pred_count
    recall = TP_count / gt_count
    f1 = 2 * (recall * precision) / (recall + precision)

    logger.info("精确率precision:%.3f,召回率recall:%.3f,F1:%.3f", precision, recall, f1)

    return precision, recall, f1


def train_check(visualizer, image, pred_boxes_scores, pred_landmarks, gts, loss, epoch, total_steps):
    logger.debug("[调试输出] Epoch[%d]，Steps[%d]，预测pred_box_socre[%r]框，预测landmark[%r], gts[%r]框，",
                 epoch, total_steps, pred_boxes_scores.shape, pred_landmarks.shape, len(gts))

    # 从tensor=>numpy(device从cuda=>cpu)
    gts = gts.cpu().detach().numpy()
    image = image.cpu().detach().numpy()
    logger.debug("画出人脸框: image[%r],pred[%r],gt[%r]", image.shape, pred_boxes_scores.shape, gts.shape)

    # GT特殊处理一下训练用的label: [N,15] [4个facebox，5个landmarks，1个置信度]，Pred不用处理，前面已经处理成box_socre和landmark了
    # GT label => box[N,4,4], ladmark[N,5]
    gt_boxes = box_utils.xywh2xyxy(gts[:, :4])  # 从label中前4位剥离box坐标
    gt_landmarks = gts[:, 4:14].reshape((-1, 5, 2))  # 从label中剥离landmark5个点
    pred_landmarks = pred_landmarks.reshape((-1, 5, 2))  # 多个人的脸的landmarks（5个点，10个值）

    draw_image = image_utils.draw(image, pred_boxes_scores, gt_boxes, pred_landmarks, gt_landmarks)

    logger.info("[可视化] 迭代[%d]steps,loss[%.4f]", total_steps, loss.item())
    visualizer.text(total_steps, loss.item(), name='train_loss')
    visualizer.image([draw_image], name="train_images")


if __name__ == '__main__':
    args = parse_argumens()
    train(args)
