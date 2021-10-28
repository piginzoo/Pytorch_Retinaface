"""
验证代码，
标签文件：用retine的widerface的验证文件，train_data/label.retina/val/label.txt
图片：train_data/images
1、解析验证用标签文件
2、通过模型进行预测
3、NMS后处理得到预测框
4、做bboxes比对得到正确率、AP等

细节：
- 可以单独运行，所以支持加载模型
- 也可以直接继承到训练过程中用于正确率验证，模型传入
- 支持调试图片生成
- 参考原作者代码，和集成部分原作者代码

AP的实现：AP，即Average PrecisionmAP，是不同置信度下的recall和precision的PR曲线下面积！
1、生成从0.01~0.99之间的置信度阈值
2、按照每个阈值，去过滤pred预测的结果，划分成正例（人脸）和负例（非人脸）
3、用所有的正例，去和GT进行IOU计算，确定，哪些预测正例是正正例，哪些GT被预测到了，从而得到recall和precision
4、用100个precision-recall/pr点，绘制出pr曲线，并计算其下面积（用差值近似），得到AP
优化：
由于计算IOU需要大量计算，可以先把所有的预测框进行IOU计算，保持一份拷贝，
然后每次从这个拷贝中，剔除哪些被置信度阈值过滤掉的预测框。

"""
import logging

import numpy as np

# from data.wider_face import WiderFaceValDataset
# from layers.functions.prior_box import PriorBox

logger = logging.getLogger(__name__)


def calc_precision_recall(iou_matrx, preds, prob_thresh):
    """
    传入一个IOU的相交矩阵（iou_matrx），行是pred的个数，列是GT的个数
    然后根据阈值（prob_thresh）确定正例，然后计算recall和precision

    recall召回率 = 我检测出的正确人脸 / 所有的人脸
    acc正确率    = 我检测出的正确人脸 / 我检测出的所有的人脸（包含错误的）

             GT1(1)  GT2(2)  GT3(1)  GT4(0)
    Pred1(1)   ️√
    Pred2(2)           ️√      ️      √
    Pred3(1)           ️√
    Pred4(0)
    这个例子说明，gt3没有和任何pred相交（未被检出），pred4页没有和任何gt相交（预测错了）

    按照pred循环，看每个pred，是不是盖上了某个gt，最终得到就得到每个gt被覆盖的情况，可以算出基于gt的recall
    然后看统计所有的pred，看可以算出基于pred的precision
    """

    # 先复制一份，因为需要修改里面的
    iou_matrx = iou_matrx.copy()

    # 通过阈值过滤出负例
    negative_indices = np.where(preds[:, 4] < prob_thresh)[0]  # 得到正例的索引们

    # 负例对应的iou_matrix的pred的行，清空所有的1=>0，这步表达的是，这个框不是正例了，相交肯定是0
    iou_matrx[negative_indices, :] = 0

    # axis=1是计算对每行求sum,很诡异,test出来的
    logger.debug("精度计算：TP:%d, P:%d, Precision:%.2f",
                 (iou_matrx.sum(axis=1) > 0).sum(),
                 iou_matrx.shape[0],
                 (iou_matrx.sum(axis=1) > 0).sum() / iou_matrx.shape[0])
    precision = (iou_matrx.sum(axis=1) > 0).sum() / iou_matrx.shape[0]
    recall = (iou_matrx.sum(axis=0) > 0).sum() / iou_matrx.shape[1]
    return precision, recall


def generate_pr_curve(iou_matrx, preds, thresh_num=100):
    """
    用来生成PR曲线的点
    :param iou_matrx: 根据IOU计算完的相交矩阵，行为pred，列为gt，相交1，不相交0
    :param pred: [N,5], 预测的信息，[x1,y1,x2,y2,prob]
    :param thresh_num: PR曲线的间隔点，即置信度阈值取多少个，默认是100个点
    :return
        返回的是一个precision-recall的数组[thresh_num=100,2]
    """
    precision_recall = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):
        thresh = 1 - (t + 1) / thresh_num  # 0.01 ~ 0.99的值
        positive_indices = np.where(preds[:, 4] >= thresh)[0]  # 得到正例的索引们
        if len(positive_indices) == 0:
            precision_recall[t, 0] = 0
            precision_recall[t, 1] = 0
        else:
            precision, recall = calc_precision_recall(iou_matrx, preds, thresh)
            precision_recall[t, 0] = precision  # 精确率
            precision_recall[t, 1] = recall  # 召回率
        logger.debug("阈值: %.2f, 对应 Precision: %.2f, Recall: %.2f", thresh, precision_recall[t, 0],
                     precision_recall[t, 1])
    return precision_recall


def calc_recall_precision_by_thresh(pred_info, proposal_list, pred_recall, thresh_num=100):
    """
    根据阈值，计算精确率precision和召回率recall
    @param thresh_num:
    """

    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):

        thresh = 1 - (t + 1) / thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index + 1] == 1)[0]
            pr_info[t, 0] = len(p_index)
            pr_info[t, 1] = pred_recall[r_index]
    return pr_info


def voc_ap(iou_matrx, preds):
    """
    计算
    voc-ap: Visual Object Classes Average Precision,AP是precision-recall曲线下面积
    recall: [100,1]，100个recall值
    precision: [100,1]，100个precision值
    """

    precision_recall = generate_pr_curve(iou_matrx, preds)
    precision = precision_recall[:, 0]
    recall = precision_recall[:, 1]

    # correct AP calculation
    # first append sentinel values at the end
    m_recall = np.concatenate(([0.], recall, [1.]))
    m_precision = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(m_precision.size - 1, 0, -1):
        m_precision[i - 1] = np.maximum(m_precision[i - 1], m_precision[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(m_recall[1:] != m_recall[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((m_recall[i + 1] - m_recall[i]) * m_precision[i + 1])
    return ap
