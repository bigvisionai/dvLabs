from dvlabs.dataAnalysis.objectDetection.annotations import Annotations
from dvlabs.utils import denormalize_bbox, calc_iou, get_batches, get_vid_writer, create_grid, calc_precision_recall_f1
import os
import cv2
import matplotlib.pyplot as plt


class Analyse:
    def __init__(self, gt_annos, pred_annos, img_dir):

        self.gt_annos = gt_annos
        self.pred_annos = pred_annos
        self.img_dir = img_dir

    def grid_view(self, save_dir=None, grid_size=(1, 1), resolution=(1280, 720),
                  maintain_ratio=True, filter_classes=[], iou_thres=1):

        image_names = list(self.gt_annos.keys())

        batch_size = grid_size[0] * grid_size[1]

        resize_w, resize_h = round(resolution[0] / grid_size[0]), round(resolution[1] / grid_size[1])

        vid_writer = None
        if save_dir is not None:
            vid_writer = get_vid_writer(os.path.join(save_dir, "grid_output.mp4"), 1,
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

    # Alias for grid view method
    view_mistakes = grid_view

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

    def avg_iou_per_sample(self):

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

    gt_anno = Annotations(gt_yolo_txt_path, img_path, class_file_path, "yolo").annotations
    # print(gt_anno)

    pd_anno = Annotations(pd_yolo_txt_path, img_path, class_file_path, "yolo").annotations
    # print(pd_anno)

    pt_analyser = Analyse(gt_anno, pd_anno, img_path)
    # pt_analyser.grid_view(grid_size=(3, 3), resolution=(1280, 720), filter_classes=[], iou_thres=.75, maintain_ratio=True)
    pt_analyser.avg_iou_per_sample()
    pt_analyser.evaluate_metric(0.5)
