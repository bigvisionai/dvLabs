import os
from xml.etree import ElementTree as ET
import cv2
import json


class Annotations:
    def __init__(self, annotation_path: str, imgs_dir: str, class_file: str, annotation_format: str):

        self.annotations = dict()
        self.classes = []

        self.classes = self.read_classes(class_file)

        if annotation_format == 'yolo':
            self.annotations = self.read_yolo(imgs_dir, annotation_path, self.classes)
        elif annotation_format == 'pascal-voc':
            self.annotations = self.read_pascal_voc_xml(annotation_path)
        elif annotation_format == 'coco':
            self.annotations = self.read_coco(annotation_path)
        else:
            #TODO throw exception
            print("Exception: format not supported yet")

    def read_classes(self, class_file):
        with open(class_file, 'r') as f:
            return f.read().split("\n")

    def read_yolo(self, img_dir="", annotations_dir="", class_names=[]):

        path, dirs, files = next(os.walk(img_dir))

        annotations = {}

        for img_name in files:

            if os.path.splitext(img_name)[-1] in ['.jpeg', '.jpg', '.png']:

                anon_file = os.path.splitext(img_name)[0] + '.txt'

                img, (img_height, img_width, img_depth) = self.read_img(
                    os.path.join(path, img_name)
                )

                read_objects = []

                with open(os.path.join(annotations_dir, anon_file), 'r') as f:

                    for idx, line in enumerate(f):
                        words = line.strip().split(" ")

                        read_objects.append(
                            {
                                'class': class_names[int(words[0])],
                                'cx': float(words[1]),
                                'cy': float(words[2]),
                                'w': float(words[3]),
                                'h': float(words[4]),
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
