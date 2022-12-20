from dvlabs.dataAnalysis.objectDetection.annotations import Annotations
import os

project_root = ".."

img_path = os.path.join(project_root, "examples", "resources", "images")
class_file_path = os.path.join(project_root, "examples", "resources", "class.names")

yolo_txt_path = os.path.join(project_root, "examples", "resources", "annotations", "yolo_txts")
anno = Annotations(yolo_txt_path, img_path, class_file_path, "yolo")
print(anno)

pascal_voc_xml_path = os.path.join(project_root, "examples", "resources", "annotations", "pascal_voc_xmls")
anno = Annotations(pascal_voc_xml_path, img_path, class_file_path, "pascal-voc")
print(anno)

coco_json_path = os.path.join(project_root, "examples", "resources", "annotations", "coco_jsons", "data.json")
anno = Annotations(coco_json_path, img_path, class_file_path, "coco")
print(anno)
