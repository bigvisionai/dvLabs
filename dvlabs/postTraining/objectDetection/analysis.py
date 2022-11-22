from dvlabs.dataAnalysis.objectDetection.annotations import Annotations
from dvlabs.utils import denormalize_bbox, calc_iou, resize_and_pad
import os
import cv2
import math
import numpy as np


def grid_view(gt_anno, pred_anno, images_dir, save_dir=None, grid_size=(1, 1), resolution=(1280, 720), maintain_ratio=True, classes=[],
              iou_thres=1):

    image_names = list(gt_anno.keys())

    batch = grid_size[0] * grid_size[1]

    resize_w, resize_h = round(resolution[0]/grid_size[0]), round(resolution[1]/grid_size[1])

    vid_writer = None
    if save_dir is not None:
        vid_writer = cv2.VideoWriter(os.path.join(save_dir, "grid_output.mp4"),
                                     cv2.VideoWriter_fourcc(*'mp4v'), fps=1,
                                     frameSize=(resize_w*grid_size[0], resize_h*grid_size[1]))

    init_idx = 0

    for x in range(math.ceil(len(image_names)/batch)):

        last_idx = init_idx + batch
        grid_img_names = image_names[init_idx:last_idx]
        init_idx = last_idx

        grid_imgs = []

        for img_name in grid_img_names:
            img_path = os.path.join(images_dir, img_name)

            img = cv2.imread(img_path)
            img_h, img_w, _ = img.shape

            filtered_gt, filtered_pred = filter_anno(gt_anno[img_name], pred_anno[img_name], iou_thres)

            display_anno(img, filtered_gt, (0, 255, 0), classes)

            display_anno(img, filtered_pred, (0, 255, 255), classes)

            grid_imgs.append(img)

        ver_imgs = []
        count = 0
        for y in range(int(grid_size[1])):
            row = []
            for x in range(int(grid_size[0])):
                try:
                    img = grid_imgs[count]
                except IndexError:
                    img = np.zeros(shape=[100, 100, 3], dtype=np.uint8)
                img = resize_and_pad(img, (resize_h, resize_w), maintain_ratio)
                row.append(img)
                count += 1
            hor_imgs = np.hstack(row)
            ver_imgs.append(hor_imgs)

        grid = np.vstack(ver_imgs)

        if vid_writer is not None:
            vid_writer.write(grid)
        else:
            cv2.imshow('grid', grid)
            key = cv2.waitKey(0)
            if key == 27 or key == ord('q'):
                break

    if vid_writer is not None:
        vid_writer.release()


def display_anno(img, img_anon, color=(0, 255, 0), classes=[]):

    for obj in img_anon['objects']:
        cls_name = obj['class']

        show = True

        if len(classes) is not 0:
            if cls_name not in classes:
                show = False

        if show:
            c_x = obj['cx'] * img_anon['width']
            c_y = obj['cy'] * img_anon['height']
            w = int(obj['w'] * img_anon['width'])
            h = int(obj['h'] * img_anon['height'])
            xmin = int(c_x - (w / 2))
            ymin = int(c_y - (h / 2))

            font = cv2.FONT_HERSHEY_SIMPLEX
            lbl_scale = 0.8
            c = round(max(img.shape)) * .03 * 1 / 22
            thickness = max(round(c * 2), 1)
            lbl_scale = lbl_scale * c
            ((lbl_w, lbl_h), lbl_bline) = cv2.getTextSize(cls_name, font, lbl_scale, thickness)
            # print((lbl_w, lbl_h), lbl_bline)
            lbl_box = [xmin, ymin-lbl_h-lbl_bline, lbl_w, lbl_h+lbl_bline]

            bbox = [xmin, ymin, w, h]

            cv2.rectangle(img, lbl_box, color, -1)
            cv2.rectangle(img, bbox, color, thickness)
            cv2.putText(img, cls_name, [xmin, ymin-lbl_bline], font, lbl_scale, thickness)


def filter_anno(gt_annos, pred_annos, iou_thres):

    filtered_pred_objs = []

    for idx, pred_obj in enumerate(pred_annos['objects']):
        bbox_iou = get_max_iou(pred_obj, pred_annos, gt_annos)

        if not bbox_iou > iou_thres:
            filtered_pred_objs.append(pred_obj)

    pred_annos['objects'] = filtered_pred_objs

    return gt_annos, pred_annos


def get_max_iou(obj, pred_annos, gt_annos):

    max_iou = 0

    for gt_obj in gt_annos['objects']:
        if obj['class'] == gt_obj['class']:
            pred_bbox = denormalize_bbox(obj, pred_annos['width'], pred_annos['height'])
            gt_bbox = denormalize_bbox(gt_obj, gt_annos['width'], gt_annos['height'])

            iou = calc_iou(pred_bbox, gt_bbox)
            if iou > max_iou:
                max_iou = iou

    return max_iou


if __name__ == "__main__":
    project_root = "..\..\.."
    img_path = os.path.join(project_root, "examples", "sample_dataset", "images")
    gt_yolo_txt_path = os.path.join(project_root, "examples", "sample_dataset", "gt")
    pd_yolo_txt_path = os.path.join(project_root, "examples", "sample_dataset", "preds")
    class_file_path = os.path.join(project_root, "examples", "sample_dataset", "class.names")

    gt_anno = Annotations(gt_yolo_txt_path, img_path, class_file_path, "yolo").annotations
    # print(gt_anno)

    pd_anno = Annotations(pd_yolo_txt_path, img_path, class_file_path, "yolo").annotations
    # print(pd_anno)

    # grid_view(gt_anno, pd_anno, img_path)
    grid_view(gt_anno, pd_anno, img_path, grid_size=(3, 3), resolution=(1280, 720),
              classes=[], iou_thres=.5, maintain_ratio=True)
