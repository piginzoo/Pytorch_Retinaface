from __future__ import print_function

import argparse
import datetime
import logging
import math
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data

from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from layers.modules import MultiBoxLoss
from models.retinaface import RetinaFace
from utils import init_log, get_device
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
    parser.add_argument('--train_dir', default='./train_data/images/train/',
                        help='Training dataset directory')
    parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--print_steps', default=1000, type=int, help='debug print interval')
    parser.add_argument('--stop', default=30, type=int, help='most early stop num')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--resume_net', default=None, help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
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

    dataset = WiderFaceDetection(train_dir, train_label, preproc(img_dim, rgb_mean))
    logger.info("数据集加载完毕：合计 %d 张", len(dataset))
    data_loader = iter(data.DataLoader(dataset,
                                       batch_size,
                                       shuffle=True,
                                       num_workers=num_workers,
                                       collate_fn=detection_collate))

    steps_of_epoch = math.ceil(len(dataset) / batch_size)
    logger.info("训练集批次：%d 张/批，一个epoch共有 %d 个批次", batch_size, steps_of_epoch)

    stepvalues = (cfg['decay1'] * steps_of_epoch, cfg['decay2'] * steps_of_epoch)
    step_index = 0

    if args.resume_epoch > 0:
        total_steps = args.resume_epoch * steps_of_epoch
    else:
        total_steps = 0

    logger.debug('开始训练...')

    # 创建优化器
    optimizer = torch.optim.Adam([{'params': net.parameters(), 'lr': 0.01}])

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

            if total_steps % args.print_steps ==0:
                logger.debug("Epoch/Step: %r/%r, 总Step:%r, loss[bbox/class/landmark]: %.4f,%.4f,%.4f",
                         epoch, step, total_steps, loss_l.item(), loss_c.item(), loss_landm.item())



        epoch_time = time.time()-epoch_start
        batch_time = epoch_time/steps_of_epoch

        if latest_loss < min_loss:
            logger.info("Epoch[%d] loss[%.4f] 比之前 loss[%.4f] 更低，保存模型",
                        epoch,
                        latest_loss,
                        min_loss)
            min_loss = latest_loss
            save_model(opt, epoch, model, len(trainloader), latest_loss, acc)

        # early_stopper可以帮助存基于acc的best模型
        if early_stopper.decide(acc, save_model, opt, epoch + 1, model, len(trainloader), latest_loss, acc):
            logger.info("早停导致退出：epoch[%d] acc[%.4f]", epoch + 1, acc)
            break

        logger.info("Epoch [%d] 结束可视化(保存softmax可视化)", epoch)
        total_steps = (epoch + 1) * len(trainloader)
        visualizer.text(total_steps, acc, name='test_acc')

    torch.save(net.state_dict(), save_folder + cfg['name'] + '_Final.pth')
    logger.debug("保存模型：%s", save_folder + cfg['name'] + '_Final.pth')


if __name__ == '__main__':
    args = parse_argumens()
    train()
