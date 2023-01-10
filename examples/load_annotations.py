from dvlabs.dataAnalysis.objectDetection.annotations import Annotations
import os

project_root = ".."
data_dir = os.path.join(project_root, 'examples', 'resources')

image_dir = os.path.join(data_dir, 'images')
classnames_filepath = os.path.join(data_dir, 'class.names')
yolo_annot = os.path.join(data_dir, 'annotations', 'yolo_txts')
voc_annot_dir = os.path.join(data_dir, 'annotations', 'pascal_voc_xmls')
coco_annot = os.path.join(data_dir, 'annotations', 'coco_jsons', 'data.json')

annot = Annotations()

annot.read_yolo(image_dir, yolo_annot, classnames_filepath)
annot.read_pascal(image_dir, voc_annot_dir)
annot.read_coco(image_dir, coco_annot)

print(annot.get_annotations())
print(annot.get_class_names())
print(annot.get_image_count())
print(annot.get_object_count())