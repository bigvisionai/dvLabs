from dvlabs.dataAnalysis.objectDetection.annotations import Annotations
from dvlabs.utils import denormalize_bbox, calc_iou, get_batches, get_vid_writer, create_grid
import os
import cv2


class Analyse:
    def __init__(self, gt_annos, pred_annos, img_dir):

        self.gt_annos = gt_annos
        self.pred_annos = pred_annos
        self.img_dir = img_dir

    def grid_view(self, save_dir=None, grid_size=(1, 1), resolution=(1280, 720),
                  maintain_ratio=True, classes=[], iou_thres=1):

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

                filtered_gt, filtered_pred = self.filter_anno(self.gt_annos[img_name], self.pred_annos[img_name], iou_thres)

                self.display_anno(img, filtered_gt, (0, 255, 0), classes)
                self.display_anno(img, filtered_pred, (0, 255, 255), classes)

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

    def display_anno(self, img, img_anon, color=(0, 255, 0), classes=[]):

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

    def filter_anno(self, gt_annos, pred_annos, iou_thres):

        filtered_pred_objs = []

        for idx, pred_obj in enumerate(pred_annos['objects']):
            bbox_iou = self.get_max_iou(pred_obj, pred_annos, gt_annos)

            if not bbox_iou > iou_thres:
                filtered_pred_objs.append(pred_obj)

        pred_annos['objects'] = filtered_pred_objs

        return gt_annos, pred_annos

    def get_max_iou(self, obj, pred_annos, gt_annos):

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

    pt_analyser = Analyse(gt_anno, pd_anno, img_path)
    pt_analyser.grid_view(grid_size=(3, 3), resolution=(1280, 720), classes=[], iou_thres=.75, maintain_ratio=True)
