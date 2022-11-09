import os
import cv2


def display(img, fps=24.99, lines=[], bboxes=[], pos="tl", offset=(None, None), txt_color=(0, 0, 0),
            bg_color=(0, 255, 255), bx_color=(0, 255, 0), font=cv2.FONT_HERSHEY_SIMPLEX):

    fps_str = "FPS : " + str(round(fps, 2))

    fps_scale = .8
    line_scale = 1

    c = round(max(img.shape)) * .03 * 1/22
    thickness = max(round(c*2), 1)
    fps_scale = fps_scale * c
    line_scale = line_scale * c

    l_sizes = []
    max_l_width = 0
    sum_l_height = 0

    if len(lines) is not 0:
        for line in lines:
            (l_size, name_bline) = cv2.getTextSize(line, font, line_scale, thickness)
            l_sizes.append(l_size)

            if l_size[0] > max_l_width:
                max_l_width = l_size[0]

            sum_l_height = sum_l_height + l_size[1]

    (fps_size, fps_bline) = cv2.getTextSize("FPS : " + str(999.99), font, fps_scale, thickness)

    margin = round(max(img.shape) * 0.02)

    rec_w = max(max_l_width, fps_size[0]) + margin
    rec_h = sum_l_height + fps_size[1] + margin * (len(l_sizes)+2)

    offset_x, offset_y = offset
    if not (offset_x or offset_y):
        offset_x = offset_y = round(max(img.shape) * 0.01)

        if pos == "tl":
            pass
        elif pos == "tr":
            offset_x = img.shape[1] - rec_w - offset_x
        elif pos == "bl":
            offset_y = img.shape[0] - rec_h - offset_y
        elif pos == "br":
            offset_x = img.shape[1] - rec_w - offset_x
            offset_y = img.shape[0] - rec_h - offset_y

    if len(bboxes) is not 0:
        for box in bboxes:
            cv2.rectangle(img, box, bx_color, thickness)

    nextline = offset_y

    cv2.rectangle(img, [offset_x, offset_y, rec_w, rec_h], bg_color, -1)

    x_fps = int(offset_x + (rec_w / 2) - (fps_size[0] / 2))
    y_fps = nextline + margin + fps_size[1]
    cv2.putText(img, fps_str, (x_fps, y_fps), font, fps_scale, txt_color, thickness,
                cv2.LINE_AA)
    nextline = y_fps + margin

    if len(lines) is not 0:
        for idx, line in enumerate(lines):
            x_line = int(offset_x + (rec_w / 2) - (l_sizes[idx][0] / 2))

            y_line = nextline + l_sizes[idx][1]

            cv2.putText(img, line, (x_line, y_line), font, line_scale, txt_color, thickness,
                        cv2.LINE_AA)

            nextline = y_line + margin


if __name__ == "__main__":

    images_dir = os.path.abspath("D:\\BigVision\\library-work\\examples\\images")

    image_path = os.path.join(images_dir, "000001.jpg")
    # image_path = os.path.join(images_dir, "000001 - copy.jpg")
    # image_path = os.path.join(images_dir, "000001 - Copy (2).jpg")

    img = cv2.imread(image_path)
    # img = cv2.resize(img, (250, 250))

    # display(img)
    # display(img, lines=["test line1", "line2", "line-3"], pos="br", bboxes=[[50, 50, 150, 150], [250, 250, 175, 175]])
    display(img, lines=["test line1", "line2", "line-3"], pos="br", bboxes=[[50, 50, 150, 150], [250, 250, 175, 175]],
            offset=(15, 15), txt_color=(0, 0, 0), bg_color=(0, 255, 255), bx_color=(0, 255, 0), font=cv2.FONT_HERSHEY_SIMPLEX)

    cv2.imwrite("test.jpg", img)
    cv2.imshow("test", img)

    cv2.waitKey(0)
