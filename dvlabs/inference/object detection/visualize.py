import os
import cv2


def display(img, fps=24.99, lines=["TEST NAME LONGER",], pos="tl"):

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

    (fps_size, fps_bline) = cv2.getTextSize("FPS : " + str(999.99), font, fontScale, thickness)

    offset_x = offset_y = int(max(img.shape) * 0.01)
    margin = int(max(img.shape) * 0.02)

    l_sizes = []

    max_l_width = 0
    sum_l_height = 0

    for line in lines:
        (l_size, name_bline) = cv2.getTextSize(line, font, name_scale, name_thck)
        l_sizes.append(l_size)

        if l_size[0]>max_l_width:
            max_l_width = l_size[0]

        sum_l_height = sum_l_height + l_size[1]

    rec_w = max(max_l_width, fps_size[0]) + margin
    rec_h = sum_l_height + fps_size[1] + margin * (len(l_sizes)+2)

    if pos == "tl":
        pass
    elif pos == "tr":
        offset_x = img.shape[1] - rec_w - offset_x
    elif pos == "bl":
        offset_y = img.shape[0] - rec_h - offset_y
    elif pos == "br":
        offset_x = img.shape[1] - rec_w - offset_x
        offset_y = img.shape[0] - rec_h - offset_y

    nextline = offset_y

    cv2.rectangle(img, [offset_x, offset_y, rec_w, rec_h], (0, 255, 255), -1)

    x_fps = int(offset_x + (rec_w / 2) - (fps_size[0] / 2))
    nextline = nextline + margin
    y_fps = nextline + fps_size[1]
    nextline = y_fps + margin
    cv2.putText(img, fps_str, (x_fps, y_fps), font, fontScale, color, thickness,
                cv2.LINE_AA)

    for idx, line in enumerate(lines):
        x_line = int(offset_x + (rec_w / 2) - (l_sizes[idx][0] / 2))

        y_line = nextline + l_sizes[idx][1]
        nextline = y_line + margin

        cv2.putText(img, line, (x_line, y_line), font, name_scale, color, name_thck,
                    cv2.LINE_AA)


if __name__ == "__main__":

    images_dir = os.path.abspath("D:\\BigVision\\library-work\\examples\\images")

    image_path = os.path.join(images_dir, "000001.jpg")
    # image_path = os.path.join(images_dir, "000001 - copy.jpg")
    # image_path = os.path.join(images_dir, "000001 - Copy (2).jpg")

    img = cv2.imread(image_path)
    # img = cv2.resize(img, (250, 250))

    # display(img)
    display(img, lines=["test line1", "line2", "line-3"], pos="br")

    cv2.imwrite("test.jpg", img)
    cv2.imshow("test", img)

    cv2.waitKey(0)
