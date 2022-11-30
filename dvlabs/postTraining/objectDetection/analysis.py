from dvlabs.dataAnalysis.objectDetection.annotations import Annotations
from dvlabs.dataPreparation.objectDetection.convert_annotations import to_yolo
from dvlabs.utils import denormalize_bbox, calc_iou, get_batches, get_vid_writer, create_grid, \
    calc_precision_recall_f1, combine_img_annos
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


class Analyse:
    def __init__(self, gt_annos_obj, pred_annos_obj, img_dir):

        self.gt_annos_obj = gt_annos_obj
        self.pred_annos_obj = pred_annos_obj
        self.gt_annos = gt_annos_obj.annotations
        self.pred_annos = pred_annos_obj.annotations
        self.img_dir = img_dir

    def grid_view(self, save_dir=None, grid_size=(1, 1), resolution=(1280, 720),
                  maintain_ratio=True, filter_classes=[], iou_thres=1):

        image_names = list(self.gt_annos.keys())

        batch_size = grid_size[0] * grid_size[1]

        resize_w, resize_h = round(resolution[0] / grid_size[0]), round(resolution[1] / grid_size[1])

        vid_writer = None
        if save_dir is not None:
            vid_writer = get_vid_writer(os.path.join(save_dir, "grid_output"), 1,
                                        (resize_w*grid_size[0], resize_h*grid_size[1]))

        batches = get_batches(image_names, batch_size)

        for batch in batches:
            batch_imgs = []
            for img_name in batch:
                img_path = os.path.join(self.img_dir, img_name)
                img = cv2.imread(img_path)

                filtered_gt = self.filter_anno(self.gt_annos[img_name], self.pred_annos[img_name], filter_classes,
                                               iou_thres)
                filtered_pred = self.filter_anno(self.pred_annos[img_name], self.gt_annos[img_name], filter_classes,
                                                 iou_thres)

                self.display_anno(img, filtered_gt, (0, 255, 0))
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

        image_names = list(self.gt_annos.keys())

        batch_size = grid_size[0] * grid_size[1]

        resize_w, resize_h = round(resolution[0] / grid_size[0]), round(resolution[1] / grid_size[1])

        vid_writer = None
        if save_dir is not None:
            vid_writer = get_vid_writer(os.path.join(save_dir, "grid_output"), 1,
                                        (resize_w*grid_size[0], resize_h*grid_size[1]))

        filtered_gt_annos = {}
        filtered_pred_annos = {}
        combined_mistakes_anno = {}
        filtered_image_names = []

        for img_name in image_names:
            filtered_gt = self.filter_anno(self.gt_annos[img_name], self.pred_annos[img_name], filter_classes,
                                           iou_thres)

            filtered_pred = self.filter_anno(self.pred_annos[img_name], self.gt_annos[img_name], filter_classes,
                                             iou_thres)

            if (len(filtered_gt['objects']) is not 0) or (len(filtered_pred['objects']) is not 0):
                filtered_gt_annos[img_name] = filtered_gt
                filtered_pred_annos[img_name] = filtered_pred
                filtered_image_names.append(img_name)

                # Combine mistakes annotations to one object
                combined_mistakes_anno[img_name] = combine_img_annos(filtered_gt, filtered_pred)

        # Save mistakes annotations
        if save_dir is not None:
            save_anno_dir = os.path.join(save_dir, "annotations")
            if not os.path.exists(save_anno_dir):
                os.makedirs(save_anno_dir)

            to_yolo(combined_mistakes_anno, save_anno_dir, self.gt_annos_obj.classes)

        batches = get_batches(filtered_image_names, batch_size)

        for batch in batches:
            batch_imgs = []
            for img_name in batch:
                img_path = os.path.join(self.img_dir, img_name)
                img = cv2.imread(img_path)

                filtered_gt = filtered_gt_annos[img_name]
                filtered_pred = filtered_pred_annos[img_name]

                self.display_anno(img, filtered_gt, (0, 255, 0))
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

        for obj in img_anon['objects']:
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
            ((lbl_w, lbl_h), lbl_bline) = cv2.getTextSize(obj['class'], font, lbl_scale, thickness)
            lbl_box = [xmin, ymin-lbl_h-lbl_bline, lbl_w, lbl_h+lbl_bline]

            bbox = [xmin, ymin, w, h]

            cv2.rectangle(img, lbl_box, color, -1)
            cv2.rectangle(img, bbox, color, thickness)
            cv2.putText(img, obj['class'], [xmin, ymin-lbl_bline], font, lbl_scale, thickness)

    def filter_anno(self, annos_to_filter, annos_to_compare, filter_classes, iou_thres):

        filtered_annos = annos_to_filter.copy()
        filtered_pred_objs = []

        for idx, to_filter_obj in enumerate(annos_to_filter['objects']):

            if self.filter_class(to_filter_obj['class'], filter_classes):

                max_bbox_iou = self.get_max_iou(to_filter_obj, annos_to_filter, annos_to_compare)

                if not max_bbox_iou > iou_thres:
                    filtered_pred_objs.append(to_filter_obj)

        filtered_annos['objects'] = filtered_pred_objs

        return filtered_annos

    def avg_iou_per_sample(self, save_dir=None):

        avg_IOUs = []

        image_names = list(self.gt_annos.keys())

        for img_name in image_names:
            img_gt_annos = self.gt_annos[img_name]
            img_pred_annos = self.pred_annos[img_name]

            sum_iou = 0
            samples = 0

            for idx, obj in enumerate(img_pred_annos['objects']):
                iou = self.get_max_iou(obj, img_pred_annos, img_gt_annos)
                sum_iou += iou
                samples += 1

            for idx, obj in enumerate(img_gt_annos['objects']):
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
                for img_name, iou in zip(image_names, avg_IOUs):
                    f.write(f"{img_name} {round(iou, 3)}\n")

        plt.plot(range(0, len(image_names)), avg_IOUs)
        plt.title('Average IOU per Sample')
        plt.xlabel('Samples')
        plt.ylabel('Average IOU')
        plt.show()

    def evaluate_metric(self, iou_thres):

        image_names = list(self.gt_annos.keys())

        tp = fp = fn = 0

        for img_name in image_names:

            img_tp, img_fp, img_fn, _, _, _ = self.evaluate_metric_img(img_name, iou_thres)

            tp += img_tp
            fp += img_fp
            fn += img_fn

        precision, recall, f1 = calc_precision_recall_f1(tp, fp, fn)

        print(f"TPs:{tp}, FPs:{fp}, FNs:{fn}")
        print(f"Precision:{precision}, Recall:{recall}, F1:{f1}")

        return tp, fp, fn, precision, recall, f1

    def evaluate_metric_img(self, img_name, iou_thres):
        tp = fp = fn = 0

        img_gt_annos = self.gt_annos[img_name]
        img_pred_annos = self.pred_annos[img_name]

        for idx, obj in enumerate(img_pred_annos['objects']):
            iou = self.get_max_iou(obj, img_pred_annos, img_gt_annos)

            if iou >= iou_thres:
                tp += 1
            elif iou < iou_thres:
                fp += 1

        for idx, obj in enumerate(img_gt_annos['objects']):
            iou = self.get_max_iou(obj, img_gt_annos, img_pred_annos)

            if iou == 0:
                fn += 1

        precision, recall, f1 = calc_precision_recall_f1(tp, fp, fn)

        # print(f"TPs:{tp}, FPs:{fp}, FNs:{fn}")
        # print(f"Precision:{precision}, Recall:{recall}, F1:{f1}")

        return tp, fp, fn, precision, recall, f1

    def confusion_matrix(self, iou_thres):
        tp, fp, fn, _, _, _ = self.evaluate_metric(iou_thres)

        marks = np.array([[tp, fp],
                          [fn, np.nan]])

        fig, ax = plt.subplots()
        ax.imshow(marks, cmap='Reds', interpolation="nearest")

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(2), labels=['True', 'False'])
        ax.set_yticks(np.arange(2), labels=['True', 'False'])

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(2):
            for j in range(2):
                ax.text(j, i, marks[i, j], ha="center", va="center", color="0")

        ax.set_title("Confusion Matrix")
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Actual Values ')

        fig.tight_layout()
        plt.show()

    def filter_class(self, cls_name, filter_classes):
        include_anno = True
        if len(filter_classes) is not 0:
            if cls_name not in filter_classes:
                include_anno = False
        return include_anno

    def get_max_iou(self, obj, annos1, annos2):

        max_iou = 0

        bbox1 = denormalize_bbox(obj, annos1['width'], annos1['height'])

        for gt_obj in annos2['objects']:
            if obj['class'] == gt_obj['class']:
                bbox2 = denormalize_bbox(gt_obj, annos2['width'], annos2['height'])

                iou = calc_iou(bbox1, bbox2)
                if iou > max_iou:
                    max_iou = iou

        return max_iou


if __name__ == "__main__":
    project_root = "..\..\.."
    img_path = os.path.join(project_root, "examples", "sample_dataset", "images")
    gt_yolo_txt_path = os.path.join(project_root, "examples", "sample_dataset", "gt")
    pd_yolo_txt_path = os.path.join(project_root, "examples", "sample_dataset", "preds")
    class_file_path = os.path.join(project_root, "examples", "sample_dataset", "class.names")

    gt_anno = Annotations(gt_yolo_txt_path, img_path, class_file_path, "yolo")
    # print(gt_anno)

    pd_anno = Annotations(pd_yolo_txt_path, img_path, class_file_path, "yolo")
    # print(pd_anno)

    pt_analyser = Analyse(gt_anno, pd_anno, img_path)
    # pt_analyser.grid_view(grid_size=(3, 3), resolution=(1280, 720), filter_classes=[], iou_thres=.75,
    #                       maintain_ratio=True)
    # pt_analyser.view_mistakes(grid_size=(3, 3), save_dir=project_root, resolution=(1280, 720), filter_classes=[], iou_thres=.75,
    #                           maintain_ratio=True)
    pt_analyser.avg_iou_per_sample(save_dir=project_root)
    # pt_analyser.evaluate_metric(0.5)
    # pt_analyser.confusion_matrix(0.5)
