from dvlabs.postTraining.objectDetection.detections import Detections
import os

project_root = ".."

img_path = os.path.join(project_root, "examples", "resources", "images")
class_file_path = os.path.join(project_root, "examples", "resources", "class.names")

yolo_txt_path = os.path.join(project_root, "examples", "resources", "detections", "yolo_txts")

dets = Detections(yolo_txt_path, img_path, class_file_path, "yolo")
print(dets)
