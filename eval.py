"""
验证代码，
标签文件：用retine的widerface的验证文件，data/label.retina/val/label.txt
图片：data/images
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

import numpy
import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs": numpy.get_include()}, reload_support=True)
from utils.box import box_overlaps

logger = logging.getLogger(__name__)


def calc_iou_matrix(pred, gt, iou_thresh=0.7, xywh=True):
    """
    single image evaluation
    pred: [N,5], x1,y1,x2,y2,prob，是预测的框，带着置信度
    gt:   [N,4], x1,y1,x2,y2
    xywh: 是否坐标格式是[x,y,w,h]，如果是，需要换成[x1,y1,x2,y2]格式

    """
    pred = pred.astype(np.int32)
    gt = gt.astype(np.int32)
    _pred = pred.copy()
    _gt = gt.copy()
    if xywh:
        # 把基于宽高的预测，改成左上和右下的坐标
        # x,y,w,h => x1,y1,x2,y2
        # 0,1,2,3
        _pred[:, 2] = _pred[:, 2] + _pred[:, 0]  #
        _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
        _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
        _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    # 准备俩flag标志数组，长度是pred框个数，和，gt框个数，
    # 1：表示，被别人给罩上了，0：表示，没有人罩我，当然要基于一个阈值来决定罩还是不罩
    # 计算pred和GT重叠的框
    """
    bbox_overlaps，入参：
    - boxes: (N, 4) ndarray of float
    - query_boxes: (K, 4) ndarray of float
    出参：overlaps: (N, K) ndarray of overlap between boxes and query_boxes,是一个矩阵，每个元素是两个框的IOU
    """
    import pdb; pdb.set_trace()
    overlaps = box_overlaps.bbox_overlaps(_pred[:, :4], _gt)
    # logger.debug("三方库算出的重叠矩阵：\r %r",overlaps)
    logger.debug("三方库算出的重叠矩阵：%r [Pred,GT]", overlaps.shape)

    # 先复制一份，因为需要修改里面的
    overlaps = overlaps.copy()

    iou_maxtrix = np.zeros(overlaps.shape)

    # 按照pred进行遍历
    for pred_index in range(_pred.shape[0]):

        # 得到这个pred和每个gt的相交比iou
        iou_with_gt = overlaps[pred_index]

        # 找出和某个预测框相交最大的gt，返回IOU和它的索引
        max_iou_with_gt, max_iou_with_gt_idx = iou_with_gt.max(), iou_with_gt.argmax()

        # 如果大于阈值，说明: 1.这个pred有效，2.有个GT被匹配上了
        if max_iou_with_gt >= iou_thresh:
            # gt_match是每个gt，对一个，这步，标明这个GT被某个pred罩上了
            # gt_match[max_iou_with_gt_idx] = 1
            # pred_match[pred_index] = 1
            iou_maxtrix[pred_index, max_iou_with_gt_idx] = 1

    return iou_maxtrix


def drop_iou_matrix_by_thresh(iou_matrix, preds, prob_thresh=0.5):
    """
    根据pred的置信度度，删除掉 [pred、GT 0/1相交矩阵] 中对应的行（pred是行）
    """

    # 先复制一份，因为需要修改里面的
    iou_matrix = iou_matrix.copy()

    # 通过阈值过滤出负例
    negative_indices = np.where(preds[:, 4] < prob_thresh)[0]  # 得到正例的索引们

    # 直接从相交矩阵中，删除掉那些概率低对应的行（当然是根据置信度阈值）
    iou_matrix = np.delete(iou_matrix, negative_indices, axis=0)

    return iou_matrix


def calc_precision_recall(iou_matrix):
    """
    传入一个IOU的相交矩阵（iou_matrix），行是pred的个数，列是GT的个数
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
    # axis=1是计算对每行求sum,很诡异,test出来的
    # logger.debug("精度计算：TP:%d, P:%d, Precision:%.2f",
    #              (iou_matrix.sum(axis=1) > 0).sum(),
    #              iou_matrix.shape[0],
    #              (iou_matrix.sum(axis=1) > 0).sum() / iou_matrix.shape[0])
    TruePositive_Pred = (iou_matrix.sum(axis=1) > 0).sum()
    TruePositive_GT = (iou_matrix.sum(axis=0) > 0).sum()
    assert TruePositive_Pred == TruePositive_GT, "Pred:" + str(TruePositive_Pred) + "/GT:" + str(TruePositive_GT)

    precision = TruePositive_Pred / iou_matrix.shape[0]
    recall = TruePositive_GT / iou_matrix.shape[1]
    f1 = 2 * (recall * precision) / (recall + precision)

    # 返回：TruePositive_GT：真正和GT相交的，iou_matrix.shape[0]：按照pred的置信度过滤后的个数
    return precision, recall, f1, TruePositive_GT


def generate_pr_curve(iou_matrix, preds, thresh_num=100):
    """
    用来生成PR曲线的点
    :param iou_matrix: 根据IOU计算完的相交矩阵，行为pred，列为gt，相交1，不相交0
    :param preds: [N,5], 预测的信息，[x1,y1,x2,y2,prob]
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
            iou_matrix = drop_iou_matrix_by_thresh(iou_matrix, preds, thresh)
            precision, recall, f1, tp = calc_precision_recall(iou_matrix)
            precision_recall[t, 0] = precision  # 精确率
            precision_recall[t, 1] = recall  # 召回率
        logger.debug("阈值: %.2f, 对应 Precision: %.2f, Recall: %.2f", thresh, precision_recall[t, 0],
                     precision_recall[t, 1])
    return precision_recall


def calc_recall_precision_by_thresh(pred_info, proposal_list, pred_recall, thresh_num=100):
    """
    根据阈值，计算精确率precision和召回率recall
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


def voc_ap(iou_matrix, preds):
    """
    计算
    voc-ap: Visual Object Classes Average Precision,AP是precision-recall曲线下面积
    recall: [100,1]，100个recall值
    precision: [100,1]，100个precision值
    """

    precision_recall = generate_pr_curve(iou_matrix, preds)
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
