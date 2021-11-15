import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def point_form(boxes):
    """ Convert anchor_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from anchor_box layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert anchor_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2]) / 2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    print(max_xy.shape)
    print(min_xy.shape)
    exit()
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from anchor_box layers, Shape: [num_anchors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b) # 相交的面积
    area_a = ((box_a[:, 2] - box_a[:, 0]) * # A面积
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) * # B面积
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i)


def matrix_iof(a, b):
    """
    return iof of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)


def match(threshold, truths, anchors, variances, labels, landms, loc_t, conf_t, landm_t, idx):
    """Match each anchor box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, 4].
        anchors: (tensor) Prior boxes from anchor_box layers, Shape: [n_anchors,4].
        variances: (tensor) Variances corresponding to each anchor coord,
            Shape: [num_anchors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        landms: (tensor) Ground truth landms, Shape [num_obj, 10].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        landm_t: (tensor) Tensor to be filled w/ endcoded landm targets.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location 2)confidence 3)landm preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(anchors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best anchor for each ground truth
    best_anchor_overlap, best_anchor_idx = overlaps.max(1, keepdim=True)

    # ignore hard gt
    valid_gt_idx = best_anchor_overlap[:, 0] >= 0.2
    best_anchor_idx_filter = best_anchor_idx[valid_gt_idx, :]
    if best_anchor_idx_filter.shape[0] <= 0:
        loc_t[idx] = 0
        conf_t[idx] = 0
        return

    # [1,num_anchors] best ground truth for each anchor
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_anchor_idx.squeeze_(1)
    best_anchor_idx_filter.squeeze_(1)
    best_anchor_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_anchor_idx_filter, 2)  # ensure best anchor
    # TODO refactor: index  best_anchor_idx with long tensor
    # ensure every gt matches with its anchor of max overlap
    for j in range(best_anchor_idx.size(0)):  # 判别此anchor是预测哪一个boxes
        best_truth_idx[best_anchor_idx[j]] = j
    matches = truths[best_truth_idx]  # Shape: [num_anchors,4] 此处为每一个anchor对应的bbox取出来
    conf = labels[best_truth_idx]  # Shape: [num_anchors]      此处为每一个anchor对应的label取出来
    conf[best_truth_overlap < threshold] = 0  # label as background   overlap<0.35的全部作为负样本
    loc = encode(matches, anchors, variances)

    matches_landm = landms[best_truth_idx]
    landm = encode_landm(matches_landm, anchors, variances)
    loc_t[idx] = loc  # [num_anchors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_anchors] top class label for each anchor
    landm_t[idx] = landm


def encode(matched, anchors, variances):
    """Encode the variances from the anchor_box layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the anchor boxes.
    Args:
        matched: (tensor) Coords of ground truth for each anchor in point-form
            Shape: [num_anchors, 4].
        anchors: (tensor) Prior boxes in center-offset form
            Shape: [num_anchors,4].
        variances: (list[float]) Variances of anchor_boxes
    Return:
        encoded boxes (tensor), Shape: [num_anchors, 4]
    """

    # dist b/t match center and anchor's center
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - anchors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * anchors[:, 2:])
    # match wh / anchor wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / anchors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_anchors,4]


def encode_landm(matched, anchors, variances):
    """Encode the variances from the anchor_box layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the anchor boxes.
    Args:
        matched: (tensor) Coords of ground truth for each anchor in point-form
            Shape: [num_anchors, 10].
        anchors: (tensor) Prior boxes in center-offset form
            Shape: [num_anchors,4].
        variances: (list[float]) Variances of anchor_boxes
    Return:
        encoded landm (tensor), Shape: [num_anchors, 10]
    """

    # dist b/t match center and anchor's center
    matched = torch.reshape(matched, (matched.size(0), 5, 2))
    anchors_cx = anchors[:, 0].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    anchors_cy = anchors[:, 1].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    anchors_w = anchors[:, 2].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    anchors_h = anchors[:, 3].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    anchors = torch.cat([anchors_cx, anchors_cy, anchors_w, anchors_h], dim=2)
    g_cxcy = matched[:, :, :2] - anchors[:, :, :2]
    # encode variance
    g_cxcy /= (variances[0] * anchors[:, :, 2:])
    # g_cxcy /= anchors[:, :, 2:]
    g_cxcy = g_cxcy.reshape(g_cxcy.size(0), -1)
    # return target for smooth_l1_loss
    return g_cxcy


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, anchors, variances):
    """Decode locations from predictions using anchors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_anchors,4]
        anchors (tensor): Prior boxes in center-offset form.
            Shape: [num_anchors,4].
        variances: (list[float]) Variances of anchor_boxes: [0.1,0.2]
    Return:
        decoded bounding box predictions
        box[t_x,t_y,t_w,t_h]

    我理解，这个是根据预测结果，去修正anchor，
    anchor是固定大小和宽度的框，
    不同层的anchor，基本上都是之前，根据缩小的feature map反向推到原图上的位置（x，y），
    而宽度是定死的：

    Feature Pyramid
    P2 (105 × 27 × 256)
    P3 (53 × 27 × 256)
    P4 (27 × 27 × 256)

    'variance': [0.1, 0.2], ？？？为何需要一个variance?!
    """

    boxes = torch.cat((
        anchors[:, :2] + loc[:, :2] * variances[0] * anchors[:, 2:],
        anchors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)

    xywh2xyxy(boxes)
    return boxes

def xywh2xyxy(boxes):
    """
    把预测出来的xywh，转化成x1y1,x2y2结果
    :param boxes: shape [N,4]
    :return:
    """

    # box是[x,y,w,h] => [x1,y1,x2,y2]
    # 左上是[x-w/2, y-h/w]，右下角=左上角+[w,h]
    # 前2个，左上角，x=x-w/2，y=y-h/2
    logger.debug("boxes.shape:%r",boxes.shape)
    boxes[:, :2] -= boxes[:, 2:] / 2
    # 后2个，右下角，x=x+w/2，y=y+h/2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre, anchors, variances):
    """Decode landm from predictions using anchors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_anchors,10]
        anchors (tensor): Prior boxes in center-offset form.
            Shape: [num_anchors,4].
        variances: (list[float]) Variances of anchor_boxes
    Return:
        decoded landm predictions
    """
    landms = torch.cat((anchors[:, :2] + pre[:, :2] * variances[0] * anchors[:, 2:],
                        anchors[:, :2] + pre[:, 2:4] * variances[0] * anchors[:, 2:],
                        anchors[:, :2] + pre[:, 4:6] * variances[0] * anchors[:, 2:],
                        anchors[:, :2] + pre[:, 6:8] * variances[0] * anchors[:, 2:],
                        anchors[:, :2] + pre[:, 8:10] * variances[0] * anchors[:, 2:],
                        ), dim=1)
    return landms


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_anchors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_anchors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_anchors.
    """

    keep = torch.Tensor(scores.size(0)).fill_(0).long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count
