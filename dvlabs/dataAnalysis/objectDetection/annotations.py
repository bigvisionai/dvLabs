import os
from xml.etree import ElementTree as ET
import cv2
import json
from dvlabs.config import annotation_formats, image_extensions, yolo_bb_format, lib_annotation_format
from dvlabs.utils import read_lines, read_yolo_annotation_txt, img_width_height_channels


class Annotations:
    def __init__(self, annotation_path: str, imgs_dir: str, class_file: str, annotation_format: str):

        assert annotation_format in annotation_formats.list(), f"{annotation_format} " \
                                                               f"is not supported. Supported annotations " \
                                                               f"formats are: {annotation_formats.list}"

        self.annotations = dict()

        self.classes = read_lines(class_file)

        if annotation_format == annotation_formats.YOLO:
            self.annotations = self.read_yolo(imgs_dir, annotation_path, self.classes)
        elif annotation_format == annotation_formats.VOC:
            self.annotations = self.read_pascal_voc_xml(annotation_path)
        elif annotation_format == annotation_formats.COCO:
            self.annotations = self.read_coco(annotation_path)

    def read_yolo(self, img_dir="", annotations_dir="", class_names=[]):

        path, dirs, files = next(os.walk(img_dir))

        annotations = {}

        for img_name in files:

            if os.path.splitext(img_name)[-1] in image_extensions.list:

                # Annotation filename corresponds to the image filename.
                anon_file = os.path.splitext(img_name)[0] + '.txt'

                img_path = os.path.join(path, img_name)
                try:
                    img_height, img_width, img_channels = img_width_height_channels(img_path)
                except Exception as e:
                    print(f"Dropping image: {img_path}.\n{e}")
                    continue

                read_objects = []

                annotation_file_path = os.path.join(annotations_dir, anon_file)

                try:
                    bboxes = read_yolo_annotation_txt(annotation_file_path)
                except Exception as e:
                    print(f'Dropping image: {img_path}. \n{e}')
                    

                for bbox in bboxes:
                    read_objects.append(
                        {
                            yolo_bb_format.CLASS: class_names[int(bbox[0])],
                            yolo_bb_format.CX: float(bbox[1]),
                            yolo_bb_format.CY: float(bbox[2]),
                            yolo_bb_format.W: float(bbox[3]),
                            yolo_bb_format.H: float(bbox[4]),
                        }
                    )

                annotations[img_name] = {
                    lib_annotation_format.IMG_WIDTH: img_width,
                    lib_annotation_format.IMG_HEIGHT: img_height,
                    lib_annotation_format.IMG_DEPTH: img_channels,
                    lib_annotation_format.CLASS_NAMES: self.classes,
                    lib_annotation_format.OBJECTS: read_objects
                }

        return annotations

    def read_pascal_voc_xml(self, annotations_dir=""):

        path, dirs, files = next(os.walk(annotations_dir))

        annotations = {}

        for file in files:

            if os.path.splitext(file)[-1] == ".xml":
                root = ET.parse(os.path.join(path, file)).getroot()

                img_name = root.find('filename').text

                size = root.find('size')
                img_width = int(size.find('width').text)
                img_height = int(size.find('height').text)
                img_depth = int(size.find('depth').text)

                objects = root.findall('object')

                read_objects = []

                for idx, object in enumerate(objects):
                    class_name = object.find('name').text

                    bbox = object.find('bndbox')
                    box_left = int(bbox.find('xmin').text)
                    box_top = int(bbox.find('ymin').text)
                    box_right = int(bbox.find('xmax').text)
                    box_bottom = int(bbox.find('ymax').text)

                    box_width, box_height = box_right - box_left, box_bottom - box_top

                    norm_anno = self.normalize_annotations(img_width, img_height, box_left, box_top, box_width, box_height)

                    read_objects.append(
                        {
                            'class': class_name,
                            'cx': norm_anno[0],
                            'cy': norm_anno[1],
                            'w': norm_anno[2],
                            'h': norm_anno[3],
                        }
                    )

                annotations[img_name] = {
                    'width': img_width,
                    'height': img_height,
                    'depth': img_depth,
                    'classes': self.classes,
                    'objects': read_objects
                }

        return annotations

    def read_coco(self, json_file_path=""):

        with open(json_file_path, 'r') as json_file:
            root = json.load(json_file)

            annotations = {}

            class_names = {}
            for category in root['categories']:
                class_names[category['id']] = category['name']

            img_id_map = {}
            for image in root['images']:
                img_id_map[image['id']] = image['file_name']

                annotations[image['file_name']] = {
                    'width': image['width'],
                    'height': image['height'],
                    'classes': self.classes,
                    'objects': []
                }

            for object in root['annotations']:
                img_name = img_id_map[object['image_id']]
                class_name = class_names[object['category_id']]

                box_left = object['bbox'][0]
                box_top = object['bbox'][1]
                box_width = object['bbox'][2]
                box_height = object['bbox'][3]

                img_width, img_height = annotations[img_name]['width'], annotations[img_name]['height'],

                norm_anno = self.normalize_annotations(img_width, img_height, box_left, box_top, box_width, box_height)

                annotations[img_name]['objects'].append(
                    {
                        'class': class_name,
                        'cx': norm_anno[0],
                        'cy': norm_anno[1],
                        'w': norm_anno[2],
                        'h': norm_anno[3],
                    }
                )

        return annotations

    def normalize_annotations(self, img_width, img_height, box_left, box_top, box_width, box_height):
        dec_places = 2

        center_x_ratio = round((box_left + int(box_width / 2)) / img_width, dec_places)
        center_y_ratio = round((box_top + int(box_height / 2)) / img_height, dec_places)
        width_ratio = round(box_width / img_width, dec_places)
        height_ratio = round(box_height / img_height, dec_places)

        return [center_x_ratio, center_y_ratio, width_ratio, height_ratio]

    def read_img(self, img_path):
        img = cv2.imread(img_path)

        return img, img.shape

    def __str__(self):
        return str(self.annotations)
