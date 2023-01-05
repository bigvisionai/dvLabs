import os

import cv2

from dvlabs.inference.objectDetection.visualize import display

images_dir = os.path.join("resources", "images")
image_path = os.path.join(images_dir, "000001.jpg")

img = cv2.imread(image_path)
# img = cv2.resize(img, (250, 250))

# display(img)
display(img, lines=["test line1", "line2", "line-3"], pos="br", bboxes=[[50, 50, 150, 150], [250, 250, 175, 175]],
        offset=(15, 15), txt_color=(0, 0, 0), bg_color=(0, 255, 255), bx_color=(0, 255, 0),
        font=cv2.FONT_HERSHEY_SIMPLEX)

cv2.imshow("test", img)
cv2.waitKey(0)
