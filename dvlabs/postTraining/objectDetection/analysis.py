import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from dvlabs.dataPreparation.objectDetection.convert_annotations import to_yolo
from dvlabs.postTraining.objectDetection import metrics
from dvlabs.config import lib_annotation_format, yolo_bb_format, label_positions
from dvlabs.inference.objectDetection.visualize import display_anno
from dvlabs.utils import (denormalize_bbox, get_batches, get_vid_writer, create_grid, combine_img_annos,
                          check_and_create_dir, get_max_iou_with_true_label, get_colors)


class Analyse:
    def __init__(self, gt_annos_obj, pred_dets_obj):

        assert gt_annos_obj.class_names == pred_dets_obj.class_names, \
            "Classes cannot be different in ground truth and detections."

        self.gt_annos_obj = gt_annos_obj
        self.pred_dets_obj = pred_dets_obj
        self.gt_annos = gt_annos_obj.annotations
        self.pred_dets = pred_dets_obj.annotations

        self.class_names = gt_annos_obj.class_names

        self.class_colors = get_colors(len(gt_annos_obj.class_names))

    def view(self, save_dir=None, grid_size=(1, 1), resolution=(1280, 720), view_mistakes=False,
             maintain_ratio=True, filter_classes=[], iou_thres=1, show_labels=True):

        assert grid_size[0] >= 1 and grid_size[1] >= 1, "Grid dimensions cannot be less than 1."
        assert resolution[0] >= 1 and resolution[1] >= 1, "Resolution cannot be less than 1."
        assert (iou_thres >= 0) and (iou_thres <= 1), "IOU threshold must be between 0-1."
        for f_cls in filter_classes:
            assert f_cls in self.class_names, "Filter class name not present in dataset."

        resize_w, resize_h = round(resolution[0] / grid_size[0]), round(resolution[1] / grid_size[1])

        # Initialize video writer
        vid_writer = None
        if save_dir is not None:
            # Create directory if not present
            check_and_create_dir(save_dir)

            vid_name = "grid_output"

            if view_mistakes:
                vid_name = "mistakes"

            vid_writer = get_vid_writer(os.path.join(save_dir, vid_name), 1,
                                        (resize_w*grid_size[0], resize_h*grid_size[1]))

        # Filter annotations
        filtered_image_ids, filtered_gt_annos, filtered_pred_annos, combined_mistakes_anno \
            = self.filter_anno(filter_classes, iou_thres, view_mistakes)

        # Save mistakes annotations
        if view_mistakes and (save_dir is not None):
            self.save_mistakes_anno(combined_mistakes_anno, save_dir)

        # Split filtered annotations into batches based on grid size
        batch_size = grid_size[0] * grid_size[1]
        batches = get_batches(filtered_image_ids, batch_size)

        # Process batches
        batch_idx = 0
        while True:
            grid = self.process_grid_batch(batches[batch_idx], filtered_gt_annos, filtered_pred_annos, grid_size,
                                           (resize_h, resize_w), maintain_ratio, show_labels)

            # Write grid frame to video or show in window
            if vid_writer is not None:
                vid_writer.write(grid)
                if batch_idx == (len(batches) - 1):  # If last frame
                    # Release video writer
                    vid_writer.release()
                    break
                batch_idx += 1
            else:
                cv2.imshow('grid', grid)
                key = cv2.waitKey(0)
                if key == 27 or key == ord('q'):  # Esc or 'Q' to exit the grid view
                    break
                elif key == 97:  # 'A' for previous frame
                    if batch_idx > 0:
                        batch_idx -= 1
                elif key == 100:  # 'D' for next frame
                    if batch_idx < (len(batches)-1):
                        batch_idx += 1

    def filter_anno(self, filter_classes, iou_thres, view_mistakes):

        image_ids = list(self.gt_annos.keys())

        filtered_gt_annos = {}
        filtered_pred_annos = {}
        combined_mistakes_anno = {}
        filtered_image_ids = []

        for img_id in image_ids:
            filtered_gt = self.filter_img_anno(self.gt_annos[img_id], self.pred_dets[img_id], filter_classes,
                                               iou_thres)

            filtered_pred = self.filter_img_anno(self.pred_dets[img_id], self.gt_annos[img_id], filter_classes,
                                                 iou_thres)

            if view_mistakes and ((len(filtered_gt[lib_annotation_format.OBJECTS]) is 0) and
                                  (len(filtered_pred[lib_annotation_format.OBJECTS]) is 0)):
                pass
            else:
                filtered_gt_annos[img_id] = filtered_gt
                filtered_pred_annos[img_id] = filtered_pred
                filtered_image_ids.append(img_id)

                if view_mistakes:
                    # Combine mistakes annotations to one object
                    combined_mistakes_anno[img_id] = combine_img_annos(filtered_gt, filtered_pred)

        return filtered_image_ids, filtered_gt_annos, filtered_pred_annos, combined_mistakes_anno

    def filter_img_anno(self, annos_to_filter, annos_to_compare, filter_classes, iou_thres):

        filtered_annos = annos_to_filter.copy()
        filtered_pred_objs = []

        for idx, to_filter_obj in enumerate(annos_to_filter[lib_annotation_format.OBJECTS]):

            if (len(filter_classes) is 0) or ((len(filter_classes) is not 0) and
                                              (to_filter_obj[yolo_bb_format.CLASS] in filter_classes)):
                max_bbox_iou, _ = get_max_iou_with_true_label(to_filter_obj, annos_to_filter, annos_to_compare)

                if not max_bbox_iou > iou_thres:
                    filtered_pred_objs.append(to_filter_obj)

        filtered_annos[lib_annotation_format.OBJECTS] = filtered_pred_objs

        return filtered_annos

    def save_mistakes_anno(self, combined_mistakes_anno, save_dir):
        save_anno_dir = os.path.join(save_dir, "mistakes")
        check_and_create_dir(save_anno_dir)

        to_yolo(combined_mistakes_anno, save_anno_dir, self.gt_annos_obj.class_names)

    def process_grid_batch(self, batch, filtered_gt_annos, filtered_pred_annos, grid_size, resize_dim, maintain_ratio,
                           show_labels):
        batch_imgs = []
        for img_id in batch:
            # Read image
            img_path = self.gt_annos[img_id][lib_annotation_format.IMG_PATH]
            img = cv2.imread(img_path)

            # Get image annotations
            filtered_gt = filtered_gt_annos[img_id]
            filtered_pred = filtered_pred_annos[img_id]

            # Display annotations on image
            display_anno(img, filtered_gt, self.class_names, class_colors=self.class_colors, txt_color=(255, 255, 255),
                         lbl_pos=label_positions.BR, show_labels=show_labels)
            display_anno(img, filtered_pred, self.class_names, txt_color=(255, 255, 255),
                         class_colors=self.class_colors, lbl_pos=label_positions.TL, show_labels=show_labels)

            # Add to current batch image
            batch_imgs.append(img)

        # create grid of current batch
        grid = create_grid(batch_imgs, grid_size, resize_dim, maintain_ratio)

        return grid

    def avg_iou_per_sample(self, save_dir=None):

        if save_dir is not None:
            # Create directory if not present
            check_and_create_dir(save_dir)

        avg_IOUs = []

        image_ids = list(self.gt_annos.keys())

        for img_id in image_ids:
            img_gt_annos = self.gt_annos[img_id]
            img_pred_annos = self.pred_dets[img_id]

            sum_iou = 0
            samples = 0

            for idx, obj in enumerate(img_pred_annos[lib_annotation_format.OBJECTS]):
                iou, _ = get_max_iou_with_true_label(obj, img_pred_annos, img_gt_annos)
                sum_iou += iou
                samples += 1

            for idx, obj in enumerate(img_gt_annos[lib_annotation_format.OBJECTS]):
                iou, _ = get_max_iou_with_true_label(obj, img_gt_annos, img_pred_annos)
                if iou == 0:
                    samples += 1

            if samples == 0:
                avg_IOUs.append(None)
            else:
                avg_IOUs.append(sum_iou/samples)

        # Save mistakes annotations
        if save_dir is not None:
            with open(os.path.join(save_dir, "avg_iou_per_sample.txt"), 'w') as f:
                for img_id, avg_iou in zip(image_ids, avg_IOUs):
                    img_path = self.gt_annos[img_id][lib_annotation_format.IMG_PATH]
                    img_name = os.path.basename(img_path)
                    if avg_iou is not None:
                        f.write(f"{img_name} {round(avg_iou, 3)}\n")
                    else:
                        f.write(f"{img_name} {None}\n")

        plt.plot(range(0, len(image_ids)), avg_IOUs)
        plt.title('Average IOU per Sample')
        plt.xlabel('Samples')
        plt.ylabel('Average IOU')
        plt.show()

    def per_class_ap(self, iou_thres, save_dir=".", print_ap=False, plot_ap=True):

        if save_dir is not None:
            # Create directory if not present
            check_and_create_dir(save_dir)

        assert (iou_thres >= 0) and (iou_thres <= 1), "IOU threshold must be between 0-1."

        image_ids = list(self.gt_annos.keys())

        tp = []
        conf = []
        pred_cls = []
        target_cls = []

        for img_id in image_ids:

            img_gt_annos = self.gt_annos[img_id]
            img_pred_annos = self.pred_dets[img_id]

            for idx, obj in enumerate(img_pred_annos[lib_annotation_format.OBJECTS]):
                iou, target_lbl = get_max_iou_with_true_label(obj, img_pred_annos, img_gt_annos)

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

        tp, fp, p, r, f1, ap, unique_classes = metrics.ap_per_class(tp, conf, pred_cls, target_cls, plot=plot_ap,
                                                                    save_dir=save_dir, names=self.class_names,
                                                                    eps=1e-16, prefix="")

        if print_ap:
            print(f"tp:{tp}, fp:{fp}, p:{p}, r:{r}, f1:{f1}, ap:{ap}, unique_classes:{unique_classes}")

        return tp, fp, p, r, f1, ap, unique_classes

    def confusion_matrix(self, conf=0.25, iou_thres=0.45, normalize=True, save_dir=".", print_m=False, plot_m=True):

        if save_dir is not None:
            # Create directory if not present
            check_and_create_dir(save_dir)

        assert (conf >= 0) and (conf <= 1), "Confidence must be between 0-1."
        assert (iou_thres >= 0) and (iou_thres <= 1), "IOU threshold must be between 0-1."

        image_ids = list(self.gt_annos.keys())

        detections = []
        labels = []

        for img_id in image_ids:

            img_pred_annos = self.pred_dets[img_id]

            for idx, obj in enumerate(img_pred_annos[lib_annotation_format.OBJECTS]):
                temp = denormalize_bbox(obj, img_pred_annos[lib_annotation_format.IMG_WIDTH],
                                        img_pred_annos[lib_annotation_format.IMG_HEIGHT])
                temp.append(float(obj[yolo_bb_format.CONF]))
                temp.append(self.class_names.index(obj[yolo_bb_format.CLASS]))
                detections.append(temp)

            img_gt_annos = self.gt_annos[img_id]

            for idx, obj in enumerate(img_gt_annos[lib_annotation_format.OBJECTS]):
                temp = [self.class_names.index(obj[yolo_bb_format.CLASS])]
                for x in denormalize_bbox(obj, img_gt_annos[lib_annotation_format.IMG_WIDTH],
                                          img_gt_annos[lib_annotation_format.IMG_HEIGHT]):
                    temp.append(x)
                labels.append(temp)

        detections = np.array(detections)
        labels = np.array(labels)

        cnfn_m = metrics.ConfusionMatrix(len(self.class_names), conf, iou_thres)
        cnfn_m.process_batch(detections, labels)
        if print_m:
            cnfn_m.print()
        if plot_m:
            cnfn_m.plot(normalize=normalize, save_dir=save_dir, names=self.class_names)
