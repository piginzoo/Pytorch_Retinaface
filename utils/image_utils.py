import cv2
import numpy as np

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_BLACK = (0, 0, 0)
COLOR_DARK_GREEN = (34, 139, 34)
COLOR_YELLOW = (0, 255, 255)


def draw(image, preds, gts):
    for pred in enumerate(preds):
        box = pred[:4]
        score = pred[4]
        draw_rect(image, box, COLOR_RED)
        cv2.putText(image, '{.3f}'.format(score), (box[0], box[1]), color=COLOR_RED)
    for box in gts:
        draw_rect(image, box, COLOR_GREEN)


def draw_rect(image, box, color):
    box = box.astype(np.int32)
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color=color, thickness=1)
