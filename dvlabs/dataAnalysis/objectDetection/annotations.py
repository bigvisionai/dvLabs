import os
from xml.etree import ElementTree as ET
from xml.dom import minidom
from collections import defaultdict
from dvlabs.config import image_extensions, yolo_bb_format, lib_annotation_format
from dvlabs.utils import read_lines, read_yolo_annotation_txt, img_width_height_channels, \
    dict_from_json, normalize_annotations


class Annotations:
    def __init__(self):

        self.annotations = dict()
        self.class_names = []
        self.image_id = 0

    def next_id(self):
        self.image_id += 1
        return self.image_id

    def get_annotations(self):
        return self.annotations

    def get_class_names(self):
        return self.class_names

    def get_image_count(self):
        return len(self.annotations.keys())

    def get_object_count(self):
        instances = defaultdict(int)
        for key, value_dict in self.annotations.items():
            objects = value_dict[lib_annotation_format.OBJECTS]
            for dic in objects:
                instances[dic[yolo_bb_format.CLASS]] += 1
        total_count = 0

        for key, value in instances.items():
            total_count += value

        instances['TOTAL_COUNT'] = total_count
        return dict(instances)

    def write_class_names(self, file_path):
        with open(file_path, 'w') as f:
            for idx, class_name in enumerate(self.class_names):
                if idx != 0:
                    f.write("\n")
                f.write(f"{class_name}")
        return

    def read_yolo(self, img_dir, annotations_dir, class_file):
        print('Passing YOLO data...')

        class_names = read_lines(class_file)

        for cls in class_names:
            if cls not in self.class_names:
                self.class_names.append(cls)

        path, dirs, files = next(os.walk(img_dir))

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
                    continue

                for bbox in bboxes:
                    dic = {
                            yolo_bb_format.CLASS: class_names[int(bbox[0])],
                            yolo_bb_format.CX: float(bbox[1]),
                            yolo_bb_format.CY: float(bbox[2]),
                            yolo_bb_format.W: float(bbox[3]),
                            yolo_bb_format.H: float(bbox[4])
                        }

                    if len(bbox) > 5:
                        dic[yolo_bb_format.CONF] = float(bbox[5])
                    read_objects.append(dic)

                self.annotations[self.next_id()] = {
                    lib_annotation_format.IMG_WIDTH: img_width,
                    lib_annotation_format.IMG_HEIGHT: img_height,
                    lib_annotation_format.IMG_DEPTH: img_channels,
                    lib_annotation_format.IMG_PATH: img_path,
                    lib_annotation_format.OBJECTS: read_objects
                }

        return

    def to_yolo(self, save_dir, class_names_path):
        os.makedirs(save_dir, exist_ok=True)
        self.write_class_names(class_names_path)

        for img_id, values in self.annotations.items():

            img_path = values[lib_annotation_format.IMG_PATH]
            img_name = os.path.basename(img_path)
            anno_file = os.path.splitext(img_name)[0] + ".txt"

            with open(os.path.join(save_dir, anno_file), 'w') as f:

                for idx, obj in enumerate(values[lib_annotation_format.OBJECTS]):

                    class_id = self.class_names.index(obj[yolo_bb_format.CLASS])

                    if idx != 0:
                        f.write("\n")

                    f.write(f"{class_id} {obj[yolo_bb_format.CX]} {obj[yolo_bb_format.CY]} "
                            f"{obj[yolo_bb_format.W]} {obj[yolo_bb_format.H]}")

    def read_pascal_voc_xml(self, img_dir, annotations_dir):
        print('Passing Pascal VOC data...')

        class_names = []

        path, dirs, files = next(os.walk(annotations_dir))

        for file in files:

            if os.path.splitext(file)[-1] == ".xml":
                root = ET.parse(os.path.join(path, file)).getroot()
                img_name = root.find('filename').text
                img_path = os.path.join(img_dir, img_name)

                try:
                    img_height, img_width, img_depth = img_width_height_channels(img_path)
                except Exception as e:
                    print(f"Dropping image: {img_path}.\n{e}")
                    continue

                size = root.find('size')
                img_width = int(size.find('width').text)
                img_height = int(size.find('height').text)
                img_depth = int(size.find('depth').text)

                objects = root.findall('object')

                read_objects = []

                for idx, obj in enumerate(objects):
                    class_name = obj.find('name').text

                    bbox = obj.find('bndbox')
                    box_left = int(bbox.find('xmin').text)
                    box_top = int(bbox.find('ymin').text)
                    box_right = int(bbox.find('xmax').text)
                    box_bottom = int(bbox.find('ymax').text)

                    box_width, box_height = box_right - box_left, box_bottom - box_top

                    norm_anno = normalize_annotations(img_width, img_height, box_left, box_top, box_width, box_height)

                    if class_name not in class_names:
                        class_names.append(class_name)

                    dic = {
                        yolo_bb_format.CLASS: class_name,
                        yolo_bb_format.CX: norm_anno[0],
                        yolo_bb_format.CY: norm_anno[1],
                        yolo_bb_format.W: norm_anno[2],
                        yolo_bb_format.H: norm_anno[3]
                    }

                    read_objects.append(dic)

                annot_id = self.next_id()

                self.annotations[annot_id] = {
                    lib_annotation_format.IMG_WIDTH: img_width,
                    lib_annotation_format.IMG_HEIGHT: img_height,
                    lib_annotation_format.IMG_DEPTH: img_depth,
                    lib_annotation_format.IMG_PATH: img_path,
                    lib_annotation_format.OBJECTS: read_objects
                }

        for class_name in class_names:
            if class_name not in self.class_names:
                self.class_names.append(class_name)

        return

    def to_pascal(self, save_dir, class_names_path=None):
        os.makedirs(save_dir, exist_ok=True)
        if class_names_path is not None:
            self.write_class_names(class_names_path)

        root = ET.Element('annotation')

        for img_name, values in self.annotations.items():

            img_path = values[lib_annotation_format.IMG_PATH]
            img_name = os.path.basename(img_path)

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

                ET.SubElement(bbox, 'xmin').text = str(c_x - (w / 2))
                ET.SubElement(bbox, 'xmax').text = str(c_x + (w / 2))
                ET.SubElement(bbox, 'ymin').text = str(c_y - (h / 2))
                ET.SubElement(bbox, 'ymax').text = str(c_y + (h / 2))

            anno_file = os.path.splitext(img_name)[0] + ".xml"

            ## Only works in Python >= 3.9
            # tree = ET.ElementTree(root)
            # ET.indent(tree, space="\t", level=0)
            # tree.write(os.path.join(save_dir, anno_file))

            xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml()
            with open(os.path.join(save_dir, anno_file), "w") as f:
                f.write(xmlstr)

    def read_coco(self, img_dir, json_file_path):
        print('Passing COCO data...')

        root = dict_from_json(json_file_path)

        class_id_to_name = dict()
        for category in root['categories']:
            class_id_to_name[category['id']] = category['name']

        for cls_id, cls_name in class_id_to_name.items():
            if cls_name not in self.class_names:
                self.class_names.append(cls_name)

        img_id_map = dict()
        for image in root['images']:
            filename = image['file_name']
            img_path = os.path.join(img_dir, filename)
            try:
                img_height, img_width, img_channels = img_width_height_channels(img_path)
            except Exception as e:
                print(f"Dropping image: {img_path}.\n{e}")
                continue
            annot_id = self.next_id()
            img_id_map[image['id']] = annot_id

            self.annotations[annot_id] = {
                lib_annotation_format.IMG_WIDTH: img_width,
                lib_annotation_format.IMG_HEIGHT: img_height,
                lib_annotation_format.IMG_DEPTH: img_channels,
                lib_annotation_format.IMG_PATH: img_path,
                lib_annotation_format.OBJECTS: []
            }

        for obj in root['annotations']:
            annot_id = img_id_map.get(obj['image_id'], None)
            if annot_id is None:
                image_id = obj['image_id']
                print(f'image_id: {image_id} is in annotations but not in Image.\n dropping...')
                continue
                
            class_name = class_id_to_name.get(obj['category_id'], None)

            if class_name is None:
                category_id = (obj['category_id'])
                print(f'category_id: {category_id} is not in categories.\nDropping...')
                continue

            box_left = obj['bbox'][0]
            box_top = obj['bbox'][1]
            box_width = obj['bbox'][2]
            box_height = obj['bbox'][3]
            img_width = self.annotations[annot_id][lib_annotation_format.IMG_WIDTH]
            img_height = self.annotations[annot_id][lib_annotation_format.IMG_HEIGHT]
            norm_anno = normalize_annotations(img_width, img_height, box_left, box_top, box_width, box_height)
            
            bbox = {
                        yolo_bb_format.CLASS: class_name,
                        yolo_bb_format.CX: norm_anno[0],
                        yolo_bb_format.CY: norm_anno[1],
                        yolo_bb_format.W: norm_anno[2],
                        yolo_bb_format.H: norm_anno[3],
                    }
            
            self.annotations[annot_id][lib_annotation_format.OBJECTS].append(bbox)
        return

    def __str__(self):
        return str(self.annotations)


if __name__ == '__main__':
    data_dir = 'C:\\Users\\Prakash\\pc\\work\\dv_labs\\examples\\resources'
    yolo_annot = os.path.join(data_dir, 'annotations', 'yolo_txts')
    voc_annot_dir = os.path.join(data_dir, 'annotations', 'pascal_voc_xmls')
    coco_annot = os.path.join(data_dir, 'annotations', 'coco_jsons', 'data.json')
    image_dir = os.path.join(data_dir, 'images')
    classnames_filepath = os.path.join(data_dir, 'class.names')
    annot = Annotations()
    annot.read_yolo(image_dir, yolo_annot, classnames_filepath)
    annot.read_pascal_voc_xml(image_dir, voc_annot_dir)
    annot.read_coco(image_dir, coco_annot)
    annot.to_yolo('tmp', 'tmp/class.txt')
    print(annot.get_annotations())
    print(annot.get_class_names())
    print(annot.get_image_count())
    print(annot.get_object_count())

    # annot = Annotations(annotation_path=coco_annot, imgs_dir=image_dir, class_file=classnames_filepath,
    #                     annotation_format=annotation_formats.COCO)
    # print('coco')
    # print(annot)

