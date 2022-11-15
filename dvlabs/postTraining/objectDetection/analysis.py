from dvlabs.dataAnalysis.objectDetection.annotations import Annotations
import os
import cv2
import math
import numpy as np


def grid_view(gt_anno, pred_anno, images_dir, grid_size=(1, 1), resolution=(1280, 720), classes=[], iou_filter=[]):

    image_names = list(gt_anno.keys())

    batch = grid_size[0] * grid_size[1]

    resize_w, resize_h = int(resolution[0]/grid_size[0]), int(resolution[1]/grid_size[1])

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

            display_anno(img, gt_anno[img_name], (0, 255, 0), classes)

            display_anno(img, pred_anno[img_name], (0, 255, 255), classes)

            grid_imgs.append(img)

        ver_imgs = []
        count = 0
        for y in range(int(grid_size[1])):
            row = []
            for x in range(int(grid_size[0])):
                img = grid_imgs[count]
                img = cv2.resize(img, (resize_w, resize_h))
                row.append(img)
                count += 1
            hor_imgs = np.hstack(row)
            ver_imgs.append(hor_imgs)

        grid = np.vstack(ver_imgs)

        cv2.imshow('grid', grid)
        cv2.waitKey(0)


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


if __name__ == "__main__":
    project_root = "..\..\.."
    # img_path = os.path.join(project_root, "examples", "images")
    # yolo_txt_path = os.path.join(project_root, "examples", "annotations", "yolo_txts")
    # pascal_voc_xml_path = os.path.join(project_root, "examples", "annotations", "pascal_voc_xmls")
    class_file_path = os.path.join(project_root, "examples", "class.names")
    #
    # gt_anno = Annotations(yolo_txt_path, img_path, class_file_path, "yolo").annotations
    # print(gt_anno)
    #
    # pd_anno = Annotations(pascal_voc_xml_path, img_path, class_file_path, "pascal-voc").annotations
    # print(pd_anno)

    project_root = "D:\BigVision\datasets\mask detection"
    img_path = os.path.join(project_root, "images")
    pascal_voc_xml_path = os.path.join(project_root, "annotations")

    gt_anno = Annotations(pascal_voc_xml_path, img_path, class_file_path, "pascal-voc").annotations
    # print(gt_anno)

    pd_anno = Annotations(pascal_voc_xml_path, img_path, class_file_path, "pascal-voc").annotations
    # print(pd_anno)

    # grid_view(gt_anno, pd_anno, img_path, grid_size=(3, 3), classes=[], iou_filter=[])
    grid_view(gt_anno, pd_anno, img_path, grid_size=(1, 1), classes=['without_mask'], iou_filter=[])
