import logging

import cv2
import numpy as np

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_BLACK = (0, 0, 0)
COLOR_DARK_GREEN = (34, 139, 34)
COLOR_YELLOW = (0, 255, 255)


def draw(image, pred_boxes_scores, gt_boxes, pred_landmarks, gt_landmarks):
    pred_boxes = pred_boxes_scores[:, :4]
    scores = pred_boxes_scores[:, 4]
    # logger.debug("score:%r", scores)
    scores = ['{:.3f}'.format(s) for s in scores]
    draw_boxes(image, pred_boxes, COLOR_RED, scores)
    draw_boxes(image, gt_boxes, COLOR_GREEN)
    for landmarks in pred_landmarks: # 一张图里可能有多个人脸
        draw_points(image, landmarks, COLOR_RED)
    for landmarks in gt_landmarks:
        draw_points(image, landmarks, COLOR_GREEN)
    return image


logger = logging.getLogger(__name__)


def draw_boxes(image, boxes, color, texts=None):
    if texts:
        for box, text in zip(boxes, texts):
            draw_box(image, box, color, text)
    else:
        for box in boxes:
            draw_box(image, box, color)


def draw_box(image, box, color, text=None):
    box = box.astype(np.int32)
    # logger.debug("画框box: %r", box)
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color=color, thickness=1)
    if text: cv2.putText(image, text, (box[0], box[1]), color=COLOR_RED, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1)


def draw_points(image, points, color=COLOR_RED):
    for p in points:
        draw_point(image, p, color)


def draw_point(image, point, color):
    if type(point)==np.ndarray:
        # logger.debug("画点point: %r", point)
        point = tuple(np.array(point,np.int).tolist())
    cv2.circle(image, point, 1, color, 4)
