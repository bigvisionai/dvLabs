from dvlabs.dataAnalysis.objectDetection.annotations import Annotations
import os
import cv2
import math
import numpy as np


def grid_view(gt_anno, pred_anno, images_dir, save_dir=None, grid_size=(1, 1), resolution=(1280, 720), maintain_ratio=True, classes=[],
              iou_filter=[]):

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

            display_anno(img, gt_anno[img_name], (0, 255, 0), classes)

            display_anno(img, pred_anno[img_name], (0, 255, 255), classes)

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

    # grid_view(gt_anno, pd_anno, img_path)
    grid_view(gt_anno, pd_anno, img_path, save_dir=project_root, grid_size=(3, 3), resolution=(1280, 720),
              classes=['without_mask'], iou_filter=[], maintain_ratio=True)
