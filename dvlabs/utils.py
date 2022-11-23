import numpy as np
import cv2
import math


def denormalize_bbox(anno_obj, img_w, img_h):
    c_x = anno_obj['cx'] * img_w
    c_y = anno_obj['cy'] * img_h
    w = round(anno_obj['w'] * img_w)
    h = round(anno_obj['h'] * img_h)

    xmin = round(c_x - (w / 2))
    ymin = round(c_y - (h / 2))
    xmax = round(c_x + (w / 2))
    ymax = round(c_y + (h / 2))

    return [xmin, ymin, xmax, ymax]


def calc_iou(pred_bbox, gt_bbox):
    # coordinates of the area of intersection.
    ix1 = np.maximum(gt_bbox[0], pred_bbox[0])
    iy1 = np.maximum(gt_bbox[1], pred_bbox[1])
    ix2 = np.minimum(gt_bbox[2], pred_bbox[2])
    iy2 = np.minimum(gt_bbox[3], pred_bbox[3])

    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))

    area_of_intersection = i_height * i_width

    # Ground Truth dimensions.
    gt_height = gt_bbox[3] - gt_bbox[1] + 1
    gt_width = gt_bbox[2] - gt_bbox[0] + 1

    # Prediction dimensions.
    pd_height = pred_bbox[3] - pred_bbox[1] + 1
    pd_width = pred_bbox[2] - pred_bbox[0] + 1

    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection

    iou = area_of_intersection / area_of_union

    return iou


def resize_and_pad(img, size, maintain_ratio, pad_color=114):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA
    else:  # stretching image
        interp = cv2.INTER_CUBIC

    if maintain_ratio is True:
        # aspect ratio of image
        img_aspect = w / h  # if on Python 2, you might need to cast as a float: float(w)/h

        final_aspect = sw / sh

        # compute scaling and pad sizing
        if img_aspect > final_aspect:  # horizontal image
            new_w = sw
            new_h = np.round(new_w / img_aspect).astype(int)
            pad_vert = (sh-new_h)/2
            pad_top, pad_bot = abs(np.floor(pad_vert).astype(int)), abs(np.ceil(pad_vert).astype(int))
            pad_left, pad_right = 0, 0
        elif img_aspect < final_aspect:  # vertical image
            new_h = sh
            new_w = np.round(new_h * img_aspect).astype(int)
            pad_horz = (sw-new_w)/2
            pad_left, pad_right = abs(np.floor(pad_horz).astype(int)), abs(np.ceil(pad_horz).astype(int))
            pad_top, pad_bot = 0, 0
        else:  # square image
            new_h, new_w = sh, sw
            pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

        # set pad color
        if len(img.shape) is 3 and not isinstance(pad_color, (list, tuple, np.ndarray)):
            # color image but only one color provided
            pad_color = [pad_color] * 3

        # scale and pad
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                        borderType=cv2.BORDER_CONSTANT, value=pad_color)
    else:
        # scale
        scaled_img = cv2.resize(img, (sw, sh), interpolation=interp)
    return scaled_img


def get_batches(items, batch_size):

    init_idx = 0
    batches = []

    for x in range(math.ceil(len(items) / batch_size)):
        last_idx = init_idx + batch_size

        batch = items[init_idx:last_idx]

        batches.append(batch)

        init_idx = last_idx

    return batches


def get_vid_writer(out_file: str, fps: int, f_size: tuple):

    vid_writer = cv2.VideoWriter(out_file + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, f_size)

    return vid_writer


def create_grid(imgs, grid_size, resize_shape, maintain_ratio):
    ver_imgs = []
    count = 0
    for y in range(int(grid_size[1])):
        row = []
        for x in range(int(grid_size[0])):
            try:
                img = imgs[count]
            except IndexError:
                img = np.zeros(shape=[100, 100, 3], dtype=np.uint8)
            img = resize_and_pad(img, resize_shape, maintain_ratio)
            row.append(img)
            count += 1
        hor_imgs = np.hstack(row)
        ver_imgs.append(hor_imgs)

    grid = np.vstack(ver_imgs)

    return grid
