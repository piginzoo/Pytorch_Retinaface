#!/bin/bash
Date=$(date +%Y%m%d%H%M)
if [ "$1" = "stop" ]; then
    echo "停止训练"
    ps aux|grep python|grep "name retinaface"| grep -v grep|awk '{print $2}'|xargs -I {} kill -9 {}
    exit
fi

if [ "$1" = "debug" ]
then
    echo "调试模式"
    CUDA_VISIBLE_DEVICES=0 \
    python train.py \
    --name retinaface \
    --debug \
    --network resnet50 \
    --train_label ./train_data/label.retina/train/label.txt \
    --train_dir ./train_data/images/train/ \
    --save_folder model/
else
    # 这个是用于docker的训练用的entry，CUDA_VISIBLE_DEVICES=0，因为显卡始终是第一块，所以始终为0
    echo "Docker生产模式: mode=$1"
    echo "调试模式"
    CUDA_VISIBLE_DEVICES=0 \
    python train.py \
    --name retinaface \
    --mode network resnet50 \
    --train_label ./train_data/label.retina/train/label.txt \
    --train_dir ./train_data/images/train/ \
    --save_folder model>logs/console.log 2>&1
fi