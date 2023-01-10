

# Supported Object Detection Annotations

class AnnotationFormats:
    COCO = 'coco'
    VOC = 'pascal-voc'
    YOLO = 'yolo'
    list = [COCO, VOC, YOLO]


annotation_formats = AnnotationFormats()


# Supported images files extensions

class ImageExtensions:
    PNG = '.png'
    JPG = '.jpg'
    JPEG = '.jpeg'
    list = [PNG, JPG, JPEG]


image_extensions = ImageExtensions()


# Bounding box format: class, cx, cy, w, h, conf for ground truth and prediction

class YoloBboxFormat:
    CLASS = 'class'
    CX = 'cx'
    CY = 'cy'
    W = 'w'
    H = 'h'
    CONF = 'conf'
    list = [CLASS, CX, CY, W, H, CONF]


yolo_bb_format = YoloBboxFormat()


class LibAnnotationFormat:
    IMG_WIDTH = 'img_width'
    IMG_HEIGHT = 'img_height'
    IMG_DEPTH = 'img_depth'
    IMG_PATH = 'img_path'
    OBJECTS = 'objects'


lib_annotation_format = LibAnnotationFormat()


class LabelPositions:
    TL = 'tl'
    TR = 'tr'
    BL = 'bl'
    BR = 'br'


label_positions = LabelPositions()
