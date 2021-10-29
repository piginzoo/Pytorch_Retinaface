from __future__ import print_function

import argparse
import logging
import math
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data

import eval
import pred
from data import WiderFaceTrainDataset, detection_collate, preproc, cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from layers.modules import MultiBoxLoss
from models.retinaface import RetinaFace
from utils import init_log, get_device, visualizer, save_model
from utils.early_stop import EarlyStop
from utils.visualizer import TensorboardVisualizer

init_log()

logger = logging.getLogger(__name__)


def parse_argumens():
    parser = argparse.ArgumentParser(description='Retinaface Training')
    parser.add_argument('--name', default='retinaface')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--train_label', default='./train_data/label.retina/train/label.txt',
                        help='Training dataset directory')
    parser.add_argument('--val_label', default='./train_data/label.retina/val/label.txt',
                        help='Training dataset directory')
    parser.add_argument('--train_dir', default='./train_data/images/train/',
                        help='Training dataset directory')
    parser.add_argument('--save_folder', default='./model/', help='Location to save checkpoint models')

    args = parser.parse_args()
    logger.debug("参数：%r", args)
    return args


def train(args):
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    cfg = None
    if args.network == "mobile0.25":
        logger.debug("使用Mobile.Net做backbone")
        cfg = cfg_mnet
    elif args.network == "resnet50":
        logger.debug("使用Resnet50做backbone")
        cfg = cfg_re50

    rgb_mean = (104, 117, 123)  # bgr order
    num_classes = 2
    img_dim = cfg['image_size']
    batch_size = cfg['batch_size']
    max_epoch = cfg['epoch']

    num_workers = args.num_workers
    momentum = args.momentum
    weight_decay = args.weight_decay
    initial_lr = args.lr
    gamma = args.gamma
    train_dir = args.train_dir
    train_label = args.train_label
    save_folder = args.save_folder

    if args.debug:
        logger.debug(">>>>>>>>>>>>>>>>>>>>>>>>> 调试模式！！！")
        batch_size = 1
        max_epoch = 1
        num_workers = 0

    device = get_device()

    net = RetinaFace(cfg=cfg)
    net = net.to(device)

    if args.resume_net is not None:
        state_dict = torch.load(args.resume_net)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)

    cudnn.benchmark = True

    multi_box_loss = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)
    multi_box_loss = multi_box_loss.to(device)

    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.to(device)

    visualizer = TensorboardVisualizer(cfg['tboard_dir'])
    early_stopper = EarlyStop(max_retry=args.stop)

    net.train()
    logger.debug('开始加载数据集: 图片目录：%s，标签文件：%s', train_dir, train_label)

    dataset = WiderFaceTrainDataset(train_dir, train_label, preproc(img_dim, rgb_mean))
    logger.info("数据集加载完毕：合计 %d 张", len(dataset))
    data_loader = iter(data.DataLoader(dataset,
                                       batch_size,
                                       shuffle=True,
                                       num_workers=num_workers,
                                       collate_fn=detection_collate))

    steps_of_epoch = math.ceil(len(dataset) / batch_size)
    logger.info("训练集批次：%d 张/批，一个epoch共有 %d 个批次", batch_size, steps_of_epoch)

    if args.resume_epoch > 0:
        total_steps = args.resume_epoch * steps_of_epoch
    else:
        total_steps = 0

    logger.debug('开始训练...')

    # 创建优化器
    optimizer = torch.optim.Adam([{'params': net.parameters(), 'lr': 0.01}])

    # 开始训练！
    latest_loss = -1
    for epoch in range(1, max_epoch + 1):
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

            # 前向运算
            # out = (bbox[N,4,2], class[N,2], landmark[N,5,2])
            # 运行日志：完成前向计算: bbox[1, 29126, 4],class [1, 29126, 2], landmark[1, 29126, 10]
            # 29126怎么来的？很关键！
            out = net(images)
            logger.debug("完成前向计算:bbox[%r],class[%r],landmark[%s]", out[0].shape, out[1].shape, out[2].shape)

            # 反向传播
            optimizer.zero_grad()
            loss_l, loss_c, loss_landm = multi_box_loss(out, priors, labels)
            loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
            loss.backward()
            logger.debug("完成反向传播计算")
            optimizer.step()

            total_steps += 1

            # 每隔N个batch，就算一下这个批次的正确率
            if total_steps % args.print_steps == 0:
                logger.debug("Epoch/Step: %r/%r, 总Step:%r, loss[bbox/class/landmark]: %.4f,%.4f,%.4f", epoch, step,
                             total_steps, loss_l.item(), loss_c.item(), loss_landm.item())

                # 从第3个epoch，才开始记录最小loss的模型，且F1设置为0
                if epoch > 2 and latest_loss < min_loss:
                    logger.info("Step[%d] loss[%.4f] 比之前 loss[%.4f] 都低，保存模型", epoch, latest_loss, min_loss)
                    min_loss = latest_loss
                    save_model(net, epoch, total_steps, latest_loss, 0)

        logger.info("Epoch [%d] 结束，耗时 %.2f 秒", epoch, time.time() - epoch_start)

        # 做F1的计算，并可视化图片
        precision, recall, f1 = validate(args.image_dir, args.val_label)
        visualizer.text(total_steps, precision, name='Precision')
        visualizer.text(total_steps, recall, name='Recall')
        visualizer.text(total_steps, f1, name='F1')

        # early_stopper可以帮助存基于acc的best模型
        if early_stopper.decide(f1, save_model, net, epoch + 1, total_steps, latest_loss, f1):
            logger.info("早停导致退出：epoch[%d] f1[%.4f]", epoch + 1, f1)
            break

    torch.save(net.state_dict(), save_folder + cfg['name'] + '_Final.pth')
    logger.debug("保存模型：%s", save_folder + cfg['name'] + '_Final.pth')


def validate(image_dir, label_path,CFG):
    bbox_scores_landmarks, gts = pred.test(image_dir, label_path, CFG.val_batch_size, CFG.val_batch_num)
    bbox_scores = bbox_scores_landmarks[:2]
    iou_matrx = eval.calc_iou_matrix(bbox_scores, gts, 0.5, xywh=False)
    precision, recall, f1 = eval.calc_precision_recall(iou_matrx, bbox_scores, 0.5)
    return precision, recall, f1


def train_check(images, labels, loss, epoch, total_steps):
    logger.debug("[可视化] 第%d批", total_steps)
    # 从tensor=>numpy(device从cuda=>cpu)
    labels = labels.cpu().detach().numpy()
    images = images.cpu().detach().numpy()
    logger.info("[可视化] 迭代[%d]steps,loss[%.4f]", total_steps, loss.item())
    visualizer.text(total_steps, loss.item(), name='train_loss')
    visualizer.image(images, name="train_images")


if __name__ == '__main__':
    args = parse_argumens()
    train()
