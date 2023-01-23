from dvlabs.dataAnalysis.objectDetection.annotations import Annotations
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

# img_path = os.path.join(project_root, "examples", "resources", "pothole_dataset", "train", "images")
# gt_yolo_txt_path = os.path.join(project_root, "examples", "resources", "pothole_dataset", "train", "gt_labels")
# det_yolo_txt_path = os.path.join(project_root, "examples", "resources", "pothole_dataset", "train", "det_labels")
# class_file_path = os.path.join(project_root, "examples", "resources", "pothole_dataset", "classes.txt")

gt_anno = Annotations()
gt_anno.read_yolo(img_path, gt_yolo_txt_path, class_file_path)
# print(gt_anno)

pd_dets = Annotations()
pd_dets.read_yolo(img_path, det_yolo_txt_path, class_file_path)
# print(gt_anno)

pt_analyser = Analyse(gt_anno, pd_dets)

pt_analyser.view(grid_size=(1, 1), resolution=(1280, 720), view_mistakes=False, maintain_ratio=True, filter_classes=[],
                 iou_thres=1, show_labels=True, show_conf=True)

pt_analyser.avg_iou_per_sample(save_dir=project_root)

pt_analyser.per_class_ap(0.90, save_dir=project_root, print_ap=False, plot_ap=True)

pt_analyser.confusion_matrix(conf=0, iou_thres=0, normalize=True, save_dir=project_root, print_m=False, plot_m=True)
