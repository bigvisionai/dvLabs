import os
from xml.etree import ElementTree as ET
from xml.dom import minidom
import cv2
import json


def read_pascal_voc_xml(annotations_dir=""):

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

                center_x_ratio, center_y_ratio = float((box_left + int(box_width / 2)) / img_width), float(
                    (box_top + int(box_height / 2)) / img_height)
                width_ratio, height_ratio = float(box_width / img_width), float(box_height / img_height)

                read_objects.append(
                    {
                        'class': class_name,
                        'cx': center_x_ratio,
                        'cy': center_y_ratio,
                        'w': width_ratio,
                        'h': height_ratio,
                    }
                )

            annotations[img_name] = {
                'width': img_width,
                'height': img_height,
                'depth': img_depth,
                'objects': read_objects
            }

    return annotations


def read_yolo(img_dir="", annotations_dir="", img_ext="", class_names=[]):

    path, dirs, files = next(os.walk(img_dir))

    annotations = {}

    for img_name in files:

        if os.path.splitext(img_name)[-1] in ['.jpeg', '.jpg', '.png']:

            anon_file = os.path.splitext(img_name)[0] + '.txt'

            try:
                img = cv2.imread(os.path.join(path, img_name))

                img_height = img.shape[0]
                img_width = img.shape[1]
                img_depth = img.shape[2]
            except:
                img_height = None
                img_width = None

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
                'objects': read_objects
            }

    return annotations


def read_coco(json_file_path=""):

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

            center_x_ratio, center_y_ratio = float((box_left + int(box_width / 2)) / img_width), float(
                (box_top + int(box_height / 2)) / img_height)
            width_ratio, height_ratio = float(box_width / img_width), float(box_height / img_height)

            annotations[img_name]['objects'].append(
                {
                    'class': class_name,
                    'cx': center_x_ratio,
                    'cy': center_y_ratio,
                    'w': width_ratio,
                    'h': height_ratio,
                }
            )

    return annotations


def to_yolo(annotations, save_dir="", class_names=[]):

    for img_name, values in annotations.items():

        anno_file = os.path.splitext(img_name)[0] + ".txt"

        with open(os.path.join(save_dir, anno_file), 'w') as f:

            for idx, object in enumerate(values['objects']):

                class_id = class_names.index(object['class'])

                if idx != 0:
                    f.write("\n")

                f.write(f"{class_id} {object['cx']} {object['cy']} {object['w']} {object['h']}")


def to_pascal(annotations, save_dir=""):
    root = ET.Element('annotation')

    for img_name, values in annotations.items():

        ET.SubElement(root, 'folder').text = ''

        ET.SubElement(root, 'filename').text = img_name
        ET.SubElement(root, 'path').text = img_name

        source = ET.SubElement(root, 'source')
        ET.SubElement(source, 'database')

        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(values['width'])
        ET.SubElement(size, 'height').text = str(values['width'])
        ET.SubElement(size, 'depth').text = str(values['depth'])

        ET.SubElement(root, 'segmented').text = 0

        for obj in values['objects']:
            object = ET.SubElement(root, 'object')

            ET.SubElement(object, "name").text = obj['class']

            ET.SubElement(object, 'pose').text = 'unspecified'
            ET.SubElement(object, 'truncated').text = '0'
            ET.SubElement(object, 'difficult').text = '0'
            ET.SubElement(object, 'occluded').text = '0'

            bbox = ET.SubElement(object, 'bndbox')

            c_x = obj['cx'] * values['width']
            c_y = obj['cy'] * values['height']
            w = obj['w'] * values['width']
            h = obj['h'] * values['height']

            ET.SubElement(bbox, 'xmin').text = str(c_x - (w/2))
            ET.SubElement(bbox, 'xmax').text = str(c_x + (w/2))
            ET.SubElement(bbox, 'ymin').text = str(c_y - (h/2))
            ET.SubElement(bbox, 'ymax').text = str(c_y + (h/2))

        anno_file = os.path.splitext(img_name)[0] + ".xml"

        ## Only works in Python >= 3.9
        # tree = ET.ElementTree(root)
        # ET.indent(tree, space="\t", level=0)
        # tree.write(os.path.join(save_dir, anno_file))

        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml()
        with open(os.path.join(save_dir, anno_file), "w") as f:
            f.write(xmlstr)


def to_coco(annotations, save_dir=""):

    anno_out = {}

    anno_out['info'] = {
        "year": "",
        "version": "",
        "description": "",
        "contributor": "",
        "url": "",
        "date_created": ""
    }
    anno_out['licenses'] = []
    anno_out['categories'] = []
    anno_out['images'] = []
    anno_out['annotations'] = []

    class_names = []

    for idx, img_name in enumerate(annotations):

        values = annotations[img_name]

        image_dict = dict(
            id=idx,
            license=None,
            file_name=img_name,
            height=values['height'],
            width=values['width']
        )
        anno_out['images'].append(image_dict)

        for obj_idx, object in enumerate(values['objects']):

            if object['class'] in class_names:
                class_id = class_names.index(object['class'])
            else:
                class_names.append(object['class'])
                class_id = class_names.index(object['class'])

            c_x = object['cx'] * values['width']
            c_y = object['cy'] * values['height']
            w = object['w'] * values['width']
            h = object['h'] * values['height']

            x_min = c_x - (w/2)
            y_min = c_y - (w/2)

            object_dict = dict(
                id=obj_idx,
                image_id=idx,
                category_id=class_id,
                bbox=[round(x_min), round(y_min), round(w), round(h)],
                area=None,
                segmentation=[],
                iscrowd=0
            )
            anno_out['annotations'].append(object_dict)

        for idx, class_name in enumerate(class_names):
            category_dict = dict(
                id=idx,
                name=class_name,
                supercategory=None
            )
            anno_out['categories'].append(category_dict)

        anno_file = 'annotations.json'

        with open(os.path.join(save_dir, anno_file), 'w') as f:

            f.write(json.dumps(anno_out, ensure_ascii=False, indent=4))


def pascal_voc_to_yolo(annotation_dir='', save_dir='', class_names=[]):
    annotations = read_pascal_voc_xml(annotation_dir)

    to_yolo(annotations, save_dir, class_names=class_names)


def yolo_to_pascal_voc(img_dir='', annotation_dir='', save_dir='', class_names=[]):
    annotations = read_yolo(img_dir, annotation_dir, class_names=class_names)

    to_pascal(annotations, save_dir)


class_names = ["vest", "helmet"]

project_root = "..\..\.."

pascal_voc_to_yolo(os.path.join(project_root, "examples", "annotations", "pascal_voc_xmls"), os.path.join(project_root, "outputs"), class_names)

yolo_to_pascal_voc(os.path.join(project_root, "examples", "images"), os.path.join(project_root, "examples", "annotations", "yolo_txts"), os.path.join(project_root, "outputs"), class_names)


annotations = read_coco(os.path.join(project_root, "examples", "annotations", "coco_jsons", "data.json"))
to_coco(annotations, os.path.join(project_root, "outputs"))
