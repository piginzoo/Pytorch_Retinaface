import json
import logging

import matplotlib.pyplot as plt
import numpy as np

import eval
import utils

"""
这个是测试检测用的评价指标的。

用labelme，造了一个标注数据，标签有两个a,b：
a: GT, 6个，5个被覆盖，
b: pred，8个，5个被覆盖，1个达不到iou标准，2个完全不相交
所以，precision= 5/8，recall= 5/6

另外，为了测试，我们会对pred预测的框的置信度进行模拟：
- 全都是1
- 0.1 ~ 0.9的随机分布
以此来测试AP

"""

logger = logging.getLogger(__name__)


def load_labme(txt_file="test/test_labelme.json", prob=False):
    gt_polys = []
    pred_polys = []
    with open(txt_file, 'r') as f:
        data = json.loads(f.read())
        for d in data['shapes']:
            text = d['label']
            pos = d['points']

            if text == 'a':
                gt_polys.append(pos)
            else:
                pred_polys.append(pos)

    logger.debug("Pred:%d,GT:%d", len(pred_polys), len(gt_polys))

    return pred_polys, gt_polys


def main():
    gts, preds = load_labme()
    import eval
    pred_match, gt_match = eval.eval(preds, gts, iou_thresh=0.5)
    eval.voc_ap(recall, precision)


# python -m test.test_eval
if __name__ == '__main__':
    utils.init_log()

    # case1: 测试一个精确率和recall的计算，iou_matrix.shape = [5,8]，行是pred=5，列是gt=8
    iou_matrix = np.array([
        [1, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 1, 1, 0],
        [1, 1, 0, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 1, 1]])
    preds = np.zeros((5, 5))  # bbox[5个框，5个值(x1,y1,x2,y2,prob)]
    preds[:, 4] = 1  # prob都设成1
    iou_matrix = eval.drop_iou_matrix_by_thresh(iou_matrix, preds, thresh=0.5)
    precision, recall, f1, tp = eval.calc_precision_recall(iou_matrix)
    assert precision == 0.8, precision  # 1/5
    assert recall == 0.875, recall  # 1/8

    # case2: 测试iou_matrix的计算
    """
    用labelme，造了一个标注数据，标签有两个a,b：
    a: GT, 6个，5个被覆盖，
    b: pred，7个，5个被覆盖，1个达不到iou标准，1个完全不相交
    所以，precision= 5/7，recall= 5/6
    """
    preds, gts = load_labme()
    preds_probs = np.array([1, 1, 1, 1, 1, 1, 1])[:, np.newaxis]  # 7个框都是1
    preds, gts = np.array(preds), np.array(gts)
    preds = preds.reshape(-1, 4)
    gts = gts.reshape(-1, 4)
    preds = np.append(preds, preds_probs, axis=1)
    iou_matrix = eval.calc_iou_matrix(preds, gts, 0.5, xywh=False)

    logger.debug("Pred/GT相交矩阵：%r \r %r", iou_matrix.shape, iou_matrix)
    iou_matrix = eval.drop_iou_matrix_by_thresh(iou_matrix, preds, thresh=0.5)
    precision, recall, f1, tp = eval.calc_precision_recall(iou_matrix)

    assert precision == (5 / 7), precision
    assert recall == (5 / 6), recall

    # case3:测试AP
    """
    如果测试AP，bbox的probs就不能都为1了，得人为造一个随机的了
    """
    preds[:, 4] = np.random.random(7)
    ap = eval.voc_ap(iou_matrix, preds)
    logger.debug("AP值：%.4f", ap)
    # 画AP曲线
    precision_recall = eval.generate_pr_curve(iou_matrix, preds)
    plt.figure(dpi=150, figsize=(3, 3))
    print(precision_recall)
    plt.xlim([0, 1])  # 设置x、y轴的上下限
    plt.ylim([0, 1])
    plt.plot(precision_recall[:, 0], precision_recall[:, 1], label=f'AP={round(ap, 4)}')
    plt.legend()
    plt.title('PR Curve')
    plt.show()
