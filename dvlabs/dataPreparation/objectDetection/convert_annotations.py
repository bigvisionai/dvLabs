import os
from xml.etree import ElementTree as ET
from xml.dom import minidom
import json
from dvlabs.config import lib_annotation_format


def to_yolo(annotations, save_dir="", class_names=[]):

    for img_id, values in annotations.items():

        img_path = annotations[img_id][lib_annotation_format.IMG_PATH]
        img_name = os.path.basename(img_path)

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
