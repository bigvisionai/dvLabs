import os
import cv2


def display(img, fps=24.99, algo_name="TEST NAME LONGER"):

    fps_str = "FPS : " + str(round(fps, 2))

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 0)

    thickness = 2
    fontScale = .8

    name_thck = 2
    name_scale = 1

    c = img.shape[1] * .03 * 1/22
    thickness = name_thck = max(round(c*2), 1)
    fontScale = fontScale * c
    name_scale = name_scale * c

    (name_size, name_bline) = cv2.getTextSize(algo_name, font, name_scale, name_thck)
    (fps_size, fps_bline) = cv2.getTextSize("FPS : " + str(999.99), font, fontScale, thickness)
    print(fps_size)
    print(name_size, name_bline)

    offset = int(max(img.shape) * 0.01)
    margin = int(max(img.shape) * 0.02)
    nextline = offset

    rec_w = max(name_size[0], fps_size[0]) + margin
    rec_h = name_size[1] + fps_size[1] + margin*3

    x_fps = int(offset + (rec_w / 2) - (fps_size[0] / 2))
    x_name = int(offset + (rec_w / 2) - (name_size[0] / 2))

    nextline = nextline + margin
    y_fps = nextline + fps_size[1]
    nextline = y_fps + margin
    y_name = nextline + name_size[1]
    nextline = y_name + margin

    cv2.rectangle(img, [offset, offset, rec_w, rec_h], (0, 255, 255), -1)
    cv2.putText(img, fps_str, (x_fps, y_fps), font, fontScale, color, thickness,
                cv2.LINE_AA)
    cv2.putText(img, algo_name, (x_name, y_name), font, name_scale, color, name_thck,
                cv2.LINE_AA)

if __name__ == "__main__":

    images_dir = os.path.abspath("D:\\BigVision\\library-work\\examples\\images")

    image_path = os.path.join(images_dir, "000001.jpg")
    # image_path = os.path.join(images_dir, "000001 - copy.jpg")
    # image_path = os.path.join(images_dir, "000001 - Copy (2).jpg")

    img = cv2.imread(image_path)
    img = cv2.resize(img, (250, 250))

    display(img)

    cv2.imwrite("test.jpg", img)
    cv2.imshow("test", img)

    cv2.waitKey(0)
