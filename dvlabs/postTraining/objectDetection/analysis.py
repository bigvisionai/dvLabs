import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from dvlabs.dataPreparation.objectDetection.convert_annotations import to_yolo
from dvlabs.postTraining.objectDetection import metrics
from dvlabs.config import lib_annotation_format, yolo_bb_format
from dvlabs.utils import (denormalize_bbox, calc_iou, get_batches, get_vid_writer, create_grid,
                          calc_precision_recall_f1, combine_img_annos)


class Analyse:
    def __init__(self, gt_annos_obj, pred_dets_obj, img_dir):

        self.gt_annos_obj = gt_annos_obj
        self.pred_annos_obj = pred_dets_obj
        self.gt_annos = gt_annos_obj.annotations
        self.pred_dets = pred_dets_obj.annotations
        self.img_dir = img_dir

    def grid_view(self, save_dir=None, grid_size=(1, 1), resolution=(1280, 720),
                  maintain_ratio=True, filter_classes=[], iou_thres=1):

        image_ids = list(self.gt_annos.keys())

        batch_size = grid_size[0] * grid_size[1]

        resize_w, resize_h = round(resolution[0] / grid_size[0]), round(resolution[1] / grid_size[1])

        vid_writer = None
        if save_dir is not None:
            vid_writer = get_vid_writer(os.path.join(save_dir, "grid_output"), 1,
                                        (resize_w*grid_size[0], resize_h*grid_size[1]))

        batches = get_batches(image_ids, batch_size)

        for batch in batches:
            batch_imgs = []
            for img_id in batch:
                img_path = self.gt_annos[img_id][lib_annotation_format.IMG_PATH]
                img = cv2.imread(img_path)

                filtered_gt = self.filter_anno(self.gt_annos[img_id], self.pred_dets[img_id], filter_classes,
                                               iou_thres)
                filtered_pred = self.filter_anno(self.pred_dets[img_id], self.gt_annos[img_id], filter_classes,
                                                 iou_thres)

                self.display_gt(img, filtered_gt, (0, 255, 0))
                self.display_anno(img, filtered_pred, (0, 255, 255))

                batch_imgs.append(img)

            grid = create_grid(batch_imgs, grid_size, (resize_h, resize_w), maintain_ratio)

            if vid_writer is not None:
                vid_writer.write(grid)
            else:
                cv2.imshow('grid', grid)
                key = cv2.waitKey(0)
                if key == 27 or key == ord('q'):
                    break

        if vid_writer is not None:
            vid_writer.release()

    def view_mistakes(self, save_dir=None, grid_size=(1, 1), resolution=(1280, 720),
                  maintain_ratio=True, filter_classes=[], iou_thres=1):

        image_ids = list(self.gt_annos.keys())

        batch_size = grid_size[0] * grid_size[1]

        resize_w, resize_h = round(resolution[0] / grid_size[0]), round(resolution[1] / grid_size[1])

        vid_writer = None
        if save_dir is not None:
            vid_writer = get_vid_writer(os.path.join(save_dir, "grid_output"), 1,
                                        (resize_w*grid_size[0], resize_h*grid_size[1]))

        filtered_gt_annos = {}
        filtered_pred_annos = {}
        combined_mistakes_anno = {}
        filtered_image_ids = []

        for img_id in image_ids:
            filtered_gt = self.filter_anno(self.gt_annos[img_id], self.pred_dets[img_id], filter_classes,
                                           iou_thres)

            filtered_pred = self.filter_anno(self.pred_dets[img_id], self.gt_annos[img_id], filter_classes,
                                             iou_thres)

            if (len(filtered_gt[lib_annotation_format.OBJECTS]) is not 0) or \
                    (len(filtered_pred[lib_annotation_format.OBJECTS]) is not 0):
                filtered_gt_annos[img_id] = filtered_gt
                filtered_pred_annos[img_id] = filtered_pred
                filtered_image_ids.append(img_id)

                # Combine mistakes annotations to one object
                combined_mistakes_anno[img_id] = combine_img_annos(filtered_gt, filtered_pred)

        # Save mistakes annotations
        if save_dir is not None:
            save_anno_dir = os.path.join(save_dir, "annotations")
            if not os.path.exists(save_anno_dir):
                os.makedirs(save_anno_dir)

            to_yolo(combined_mistakes_anno, save_anno_dir, self.gt_annos_obj.classes)

        batches = get_batches(filtered_image_ids, batch_size)

        for batch in batches:
            batch_imgs = []
            for img_id in batch:
                img_path = self.gt_annos[img_id][lib_annotation_format.IMG_PATH]
                img = cv2.imread(img_path)

                filtered_gt = filtered_gt_annos[img_id]
                filtered_pred = filtered_pred_annos[img_id]

                self.display_gt(img, filtered_gt, (0, 255, 0))
                self.display_anno(img, filtered_pred, (0, 255, 255))

                batch_imgs.append(img)

            grid = create_grid(batch_imgs, grid_size, (resize_h, resize_w), maintain_ratio)

            if vid_writer is not None:
                vid_writer.write(grid)
            else:
                cv2.imshow('grid', grid)
                key = cv2.waitKey(0)
                if key == 27 or key == ord('q'):
                    break

        if vid_writer is not None:
            vid_writer.release()

    def display_anno(self, img, img_anon, color=(0, 255, 0)):

        for obj in img_anon[lib_annotation_format.OBJECTS]:
            c_x = obj[yolo_bb_format.CX] * img_anon[lib_annotation_format.IMG_WIDTH]
            c_y = obj[yolo_bb_format.CY] * img_anon[lib_annotation_format.IMG_HEIGHT]
            w = int(obj[yolo_bb_format.W] * img_anon[lib_annotation_format.IMG_WIDTH])
            h = int(obj[yolo_bb_format.H] * img_anon[lib_annotation_format.IMG_HEIGHT])
            xmin = int(c_x - (w / 2))
            ymin = int(c_y - (h / 2))

            font = cv2.FONT_HERSHEY_SIMPLEX
            lbl_scale = 0.8
            c = round(max(img.shape)) * .03 * 1 / 22
            thickness = max(round(c * 2), 1)
            lbl_scale = lbl_scale * c
            ((lbl_w, lbl_h), lbl_bline) = cv2.getTextSize(obj[yolo_bb_format.CLASS], font, lbl_scale, thickness)
            lbl_box = [xmin, ymin-lbl_h-lbl_bline, lbl_w, lbl_h+lbl_bline]

            bbox = [xmin, ymin, w, h]

            cv2.rectangle(img, lbl_box, color, -1)
            cv2.rectangle(img, bbox, color, thickness)
            cv2.putText(img, obj[yolo_bb_format.CLASS], [xmin, ymin-lbl_bline], font, lbl_scale, thickness)

    def display_gt(self, img, img_anon, color=(0, 255, 0)):

        for obj in img_anon[lib_annotation_format.OBJECTS]:
            c_x = obj[yolo_bb_format.CX] * img_anon[lib_annotation_format.IMG_WIDTH]
            c_y = obj[yolo_bb_format.CY] * img_anon[lib_annotation_format.IMG_HEIGHT]
            w = int(obj[yolo_bb_format.W] * img_anon[lib_annotation_format.IMG_WIDTH])
            h = int(obj[yolo_bb_format.H] * img_anon[lib_annotation_format.IMG_HEIGHT])
            xmin = int(c_x - (w / 2))
            ymin = int(c_y - (h / 2))
            xmax = int(c_x + (w / 2))
            ymax = int(c_y + (h / 2))

            font = cv2.FONT_HERSHEY_SIMPLEX
            lbl_scale = 0.8
            c = round(max(img.shape)) * .03 * 1 / 22
            thickness = max(round(c * 2), 1)
            lbl_scale = lbl_scale * c
            ((lbl_w, lbl_h), lbl_bline) = cv2.getTextSize(obj[yolo_bb_format.CLASS], font, lbl_scale, thickness)
            lbl_box = [xmax-lbl_w, ymax, lbl_w, lbl_h+lbl_bline]

            bbox = [xmin, ymin, w, h]

            cv2.rectangle(img, lbl_box, color, -1)
            cv2.rectangle(img, bbox, color, thickness)
            cv2.putText(img, obj[yolo_bb_format.CLASS], [xmax-lbl_w, ymax+lbl_h], font, lbl_scale, thickness)

    def filter_anno(self, annos_to_filter, annos_to_compare, filter_classes, iou_thres):

        filtered_annos = annos_to_filter.copy()
        filtered_pred_objs = []

        for idx, to_filter_obj in enumerate(annos_to_filter[lib_annotation_format.OBJECTS]):

            if self.filter_class(to_filter_obj[yolo_bb_format.CLASS], filter_classes):

                max_bbox_iou = self.get_max_iou(to_filter_obj, annos_to_filter, annos_to_compare)

                if not max_bbox_iou > iou_thres:
                    filtered_pred_objs.append(to_filter_obj)

        filtered_annos[lib_annotation_format.OBJECTS] = filtered_pred_objs

        return filtered_annos

    def avg_iou_per_sample(self, save_dir=None):

        avg_IOUs = []

        image_ids = list(self.gt_annos.keys())

        for img_id in image_ids:
            img_gt_annos = self.gt_annos[img_id]
            img_pred_annos = self.pred_dets[img_id]

            sum_iou = 0
            samples = 0

            for idx, obj in enumerate(img_pred_annos[lib_annotation_format.OBJECTS]):
                iou = self.get_max_iou(obj, img_pred_annos, img_gt_annos)
                sum_iou += iou
                samples += 1

            for idx, obj in enumerate(img_gt_annos[lib_annotation_format.OBJECTS]):
                iou = self.get_max_iou(obj, img_gt_annos, img_pred_annos)
                if iou == 0:
                    samples += 1

            if samples == 0:
                avg_IOUs.append(None)
            else:
                avg_IOUs.append(sum_iou/samples)

        # Save mistakes annotations
        if save_dir is not None:
            with open(os.path.join(save_dir, "avg_iou_per_sample.txt"), 'w') as f:
                for img_id, iou in zip(image_ids, avg_IOUs):
                    img_path = self.gt_annos[img_id][lib_annotation_format.IMG_PATH]
                    img_name = os.path.basename(img_path)
                    f.write(f"{img_name} {round(iou, 3)}\n")

        plt.plot(range(0, len(image_ids)), avg_IOUs)
        plt.title('Average IOU per Sample')
        plt.xlabel('Samples')
        plt.ylabel('Average IOU')
        plt.show()

    def per_class_ap(self, iou_thres):

        image_ids = list(self.gt_annos.keys())

        tp = []
        conf = []
        pred_cls = []
        target_cls = []

        for img_id in image_ids:

            img_gt_annos = self.gt_annos[img_id]
            img_pred_annos = self.pred_dets[img_id]

            for idx, obj in enumerate(img_pred_annos[lib_annotation_format.OBJECTS]):
                iou, target_lbl = self.get_max_iou_with_true_label(obj, img_pred_annos, img_gt_annos)

                if iou >= iou_thres:
                    tp.append([True])
                elif iou != 0:
                    tp.append([False])

                if iou != 0:
                    conf.append(obj[yolo_bb_format.CONF])
                    pred_cls.append(obj[yolo_bb_format.CLASS])

                    target_cls.append(target_lbl)

        tp = np.array(tp)
        conf = np.array(conf)
        pred_cls = np.array(pred_cls)
        target_cls = np.array(target_cls)

        tp, fp, p, r, f1, ap, unique_classes = metrics.ap_per_class(tp, conf, pred_cls, target_cls, plot=True,
                                                                    save_dir='.', names=self.gt_annos_obj.class_names,
                                                                    eps=1e-16, prefix="")

        # print(f"tp:{tp}, fp:{fp}, p:{p}, r:{r}, f1:{f1}, ap:{ap}, unique_classes:{unique_classes}")

        return tp, fp, p, r, f1, ap, unique_classes

    def evaluate_metric(self, iou_thres):

        image_ids = list(self.gt_annos.keys())

        tp = fp = fn = 0

        for img_id in image_ids:

            img_tp, img_fp, img_fn, _, _, _ = self.evaluate_metric_img(img_id, iou_thres)

            tp += img_tp
            fp += img_fp
            fn += img_fn

        precision, recall, f1 = calc_precision_recall_f1(tp, fp, fn)

        print(f"TPs:{tp}, FPs:{fp}, FNs:{fn}")
        print(f"Precision:{precision}, Recall:{recall}, F1:{f1}")

        return tp, fp, fn, precision, recall, f1

    def evaluate_metric_img(self, img_id, iou_thres):
        tp = fp = fn = 0

        img_gt_annos = self.gt_annos[img_id]
        img_pred_annos = self.pred_dets[img_id]

        for idx, obj in enumerate(img_pred_annos[lib_annotation_format.OBJECTS]):
            iou = self.get_max_iou(obj, img_pred_annos, img_gt_annos)

            if iou >= iou_thres:
                tp += 1
            elif iou < iou_thres:
                fp += 1

        for idx, obj in enumerate(img_gt_annos[lib_annotation_format.OBJECTS]):
            iou = self.get_max_iou(obj, img_gt_annos, img_pred_annos)

            if iou == 0:
                fn += 1

        precision, recall, f1 = calc_precision_recall_f1(tp, fp, fn)

        # print(f"TPs:{tp}, FPs:{fp}, FNs:{fn}")
        # print(f"Precision:{precision}, Recall:{recall}, F1:{f1}")

        return tp, fp, fn, precision, recall, f1

    def confusion_matrix(self, conf=0.25, iou_thres=0.45, print_m=False, plot_m=True):
        image_ids = list(self.gt_annos.keys())

        detections = []
        labels = []

        for img_id in image_ids:

            img_pred_annos = self.pred_dets[img_id]

            for idx, obj in enumerate(img_pred_annos[lib_annotation_format.OBJECTS]):
                temp = denormalize_bbox(obj, img_pred_annos[lib_annotation_format.IMG_WIDTH],
                                        img_pred_annos[lib_annotation_format.IMG_HEIGHT])
                temp.append(float(obj[yolo_bb_format.CONF]))
                temp.append(self.pred_annos_obj.class_names.index(obj[yolo_bb_format.CLASS]))
                detections.append(temp)

            img_gt_annos = self.gt_annos[img_id]

            for idx, obj in enumerate(img_gt_annos[lib_annotation_format.OBJECTS]):
                temp = [self.gt_annos_obj.class_names.index(obj[yolo_bb_format.CLASS])]
                for x in denormalize_bbox(obj, img_gt_annos[lib_annotation_format.IMG_WIDTH],
                                          img_gt_annos[lib_annotation_format.IMG_HEIGHT]):
                    temp.append(x)
                labels.append(temp)

        detections = np.array(detections)
        labels = np.array(labels)

        cnfn_m = metrics.ConfusionMatrix(len(self.gt_annos_obj.class_names), conf, iou_thres)
        cnfn_m.process_batch(detections, labels)
        if print_m:
            cnfn_m.print()
        if plot_m:
            cnfn_m.plot()

    def filter_class(self, cls_name, filter_classes):
        include_anno = True
        if len(filter_classes) is not 0:
            if cls_name not in filter_classes:
                include_anno = False
        return include_anno

    def get_max_iou(self, obj, annos1, annos2):

        max_iou = 0

        bbox1 = denormalize_bbox(obj, annos1[lib_annotation_format.IMG_WIDTH], annos1[lib_annotation_format.IMG_HEIGHT])

        for gt_obj in annos2[lib_annotation_format.OBJECTS]:
            if obj[yolo_bb_format.CLASS] == gt_obj[yolo_bb_format.CLASS]:
                bbox2 = denormalize_bbox(gt_obj, annos2[lib_annotation_format.IMG_WIDTH],
                                         annos2[lib_annotation_format.IMG_HEIGHT])

                iou = calc_iou(bbox1, bbox2)
                if iou > max_iou:
                    max_iou = iou

        return max_iou

    def get_max_iou_with_true_label(self, obj, annos1, annos2):

        max_iou = 0
        true_lbl = None

        bbox1 = denormalize_bbox(obj, annos1[lib_annotation_format.IMG_WIDTH], annos1[lib_annotation_format.IMG_HEIGHT])

        for gt_obj in annos2[lib_annotation_format.OBJECTS]:
            bbox2 = denormalize_bbox(gt_obj, annos2[lib_annotation_format.IMG_WIDTH],
                                     annos2[lib_annotation_format.IMG_HEIGHT])

            iou = calc_iou(bbox1, bbox2)
            if iou > max_iou:
                max_iou = iou
                true_lbl = gt_obj[yolo_bb_format.CLASS]

        return max_iou, true_lbl
