import os
from dvlabs.dataAnalysis.objectDetection.annotations import Annotations
from dvlabs.dataPreparation.objectDetection.convert_annotations import to_yolo

class_names = ["vest", "helmet"]

project_root = ".."
img_path = os.path.join(project_root, "examples", "resources", "images")

coco_json_path = os.path.join(project_root, "examples", "resources", "annotations", "coco_jsons", "data.json")
class_file_path = os.path.join(project_root, "examples", "resources", "class.names")

anno = Annotations(coco_json_path, img_path, class_file_path, "coco")
print(anno.annotations)
to_yolo(anno.annotations, os.path.join(project_root, "outputs"), ["Workers", "head", "helmet", "Workers", "person"])
