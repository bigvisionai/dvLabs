from dvlabs.dataAnalysis.objectDetection.annotations import Annotations
from dvlabs.postTraining.objectDetection.detections import Detections
from dvlabs.postTraining.objectDetection.analysis import Analyse
import os

project_root = ".."
# img_path = os.path.join(project_root, "examples", "resources", "trash_dataset", "images")
# gt_yolo_txt_path = os.path.join(project_root, "examples", "resources", "trash_dataset", "gt")
# det_yolo_txt_path = os.path.join(project_root, "examples", "resources", "trash_dataset", "preds")
# class_file_path = os.path.join(project_root, "examples", "resources", "trash_dataset", "class.names")

img_path = os.path.join(project_root, "examples", "resources", "coco128", "images")
gt_yolo_txt_path = os.path.join(project_root, "examples", "resources", "coco128", "gt")
det_yolo_txt_path = os.path.join(project_root, "examples", "resources", "coco128", "gt_dets")
class_file_path = os.path.join(project_root, "examples", "resources", "coco128", "class.names")

gt_anno = Annotations(gt_yolo_txt_path, img_path, class_file_path, "yolo")
# print(gt_anno)

pd_dets = Detections(det_yolo_txt_path, img_path, class_file_path, "yolo")
# print(pd_dets)

pt_analyser = Analyse(gt_anno, pd_dets, img_path)
# pt_analyser.grid_view(grid_size=(3, 3), resolution=(1280, 720), filter_classes=[], iou_thres=.75,
#                       maintain_ratio=True)
# pt_analyser.view_mistakes(grid_size=(3, 3), resolution=(1280, 720), filter_classes=[], iou_thres=.75,
#                           maintain_ratio=True)
# pt_analyser.avg_iou_per_sample(save_dir=project_root)
pt_analyser.per_class_ap(0.90)
# pt_analyser.evaluate_metric(0.5)
pt_analyser.confusion_matrix(conf=0, iou_thres=0, print_m=False, plot_m=True)
