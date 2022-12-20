import os
import cv2


class Detections:
    def __init__(self, annotation_path: str, imgs_dir: str, class_file: str, annotation_format: str = "yolo"):

        self.detections = dict()
        self.classes = []

        self.classes = self.read_classes(class_file)

        if annotation_format == 'yolo':
            self.detections = self.read_yolo(imgs_dir, annotation_path, self.classes)
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
                                'conf': float(words[5])
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

    def read_img(self, img_path):
        img = cv2.imread(img_path)

        return img, img.shape

    def __str__(self):
        return str(self.detections)


if __name__ == '__main__':
    project_root = "..\..\.."
    img_path = os.path.join(project_root, "examples", "images")
    yolo_txt_path = os.path.join(project_root, "examples", "detections", "yolo_txts")
    class_file_path = os.path.join(project_root, "examples", "class.names")

    anno = Detections(yolo_txt_path, img_path, class_file_path, "yolo")
    print(anno)
