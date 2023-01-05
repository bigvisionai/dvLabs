import os

import cv2

from dvlabs.utils import get_font_scale_n_thickness


def display(img, fps=24.99, lines=[], bboxes=[], pos="tl", offset=(None, None), txt_color=(0, 0, 0),
            bg_color=(0, 255, 255), bx_color=(0, 255, 0), font=cv2.FONT_HERSHEY_SIMPLEX):
    """
    Display detected bounding boxes, fps, and addition lines of information in a box.
    :param img: input image
    :param fps: fps value
    :param lines: list of text lines to write
    :param bboxes: list of bounding boxes to draw
    :param pos: position of info box ('tl', 'tr', 'bl', 'br')
    :param offset: offset of info box from top-left cornet (this overrides the pos argument)
    :param txt_color: color of text
    :param bg_color: color of info box
    :param bx_color: color of bounding box
    :param font: text font
    :return:
    """

    # Define FPS line
    fps_str = "FPS : {}"

    # Get scale and thickness of fonts
    fps_scale, thickness = get_font_scale_n_thickness(img.shape, scale_factor=0.8)
    line_scale, _ = get_font_scale_n_thickness(img.shape)

    # Get sizes of text lines
    l_sizes, max_l_width, sum_l_height = get_line_sizes(lines, font, line_scale, thickness)

    # Get maximum possible size of fps line
    (fps_size, fps_bline) = cv2.getTextSize(fps_str.format(999.99), font, fps_scale, thickness)

    # Define margin for text in info box
    margin = round(max(img.shape) * 0.02)

    # Get info box dimensions
    rec_w = max(max_l_width, fps_size[0]) + margin
    rec_h = sum_l_height + fps_size[1] + margin * (len(l_sizes)+2)

    # Get offset if explicitly mentioned. If not, calculate based on position of info box
    offset_x, offset_y = offset
    if not (offset_x or offset_y):
        offset_x, offset_y = get_offsets(pos, img.shape, rec_w, rec_h)

    # Draw detected boxes
    if len(bboxes) is not 0:
        draw_bboxes(img, bboxes, bx_color, thickness)

    # Updated y offset for initial line
    nextline = offset_y

    # Draw info box
    cv2.rectangle(img, [offset_x, offset_y, rec_w, rec_h], bg_color, -1)

    # Calculate position for fps line
    x_fps = int(offset_x + (rec_w / 2) - (fps_size[0] / 2))
    y_fps = nextline + margin + fps_size[1]

    # Write fps line
    cv2.putText(img, fps_str.format(round(fps, 2)), (x_fps, y_fps), font, fps_scale, txt_color, thickness, cv2.LINE_AA)

    # Update y offset for next line
    nextline = y_fps + margin

    # Write lines
    if len(lines) is not 0:
        write_lines(img, lines, l_sizes, offset_x, nextline, margin, rec_w, font, line_scale, txt_color, thickness)


def get_line_sizes(lines, font, line_scale, thickness):
    """
    Calculate sizes of the text lines, maximum width of text and sum of heights of line texts
    :param lines:
    :param font:
    :param line_scale:
    :param thickness:
    :return:
    """
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

    return l_sizes, max_l_width, sum_l_height


def draw_bboxes(img, bboxes, color, thickness):
    """
    Draws bounding boxes on image
    :param img: image to draw boxes on
    :param bboxes: list of bounding boxes
    :param color: color of box
    :param thickness: thickness of box
    :return:
    """
    for box in bboxes:
        cv2.rectangle(img, box, color, thickness)


def write_lines(img, lines, l_sizes, offset_x, nextline, margin, rec_w, font, line_scale, txt_color, thickness):
    """
    Write lines on the image
    :param img: image to write lines on
    :param lines: list of lines
    :param l_sizes: sizes of all the lines
    :param offset_x: x offset of info box
    :param nextline: y coordinate of next line
    :param margin: margin for info box
    :param rec_w: width of info box
    :param font: text font
    :param line_scale: scale for line font
    :param txt_color: text color
    :param thickness: text thickness
    :return:
    """
    for idx, line in enumerate(lines):
        # Calculate position for text line
        x_line = int(offset_x + (rec_w / 2) - (l_sizes[idx][0] / 2))
        y_line = nextline + l_sizes[idx][1]

        # Write line
        cv2.putText(img, line, (x_line, y_line), font, line_scale, txt_color, thickness, cv2.LINE_AA)

        # Update y offset for next line
        nextline = y_line + margin


def get_offsets(pos: str, img_shape: tuple, rec_w: int, rec_h: int) -> (int, int):
    """
    Calculate x,y offsets based on the position of the info box
    :param pos: position of info box ('tl', 'tr', 'bl', 'br')
    :param img_shape: shape of image
    :param rec_w: width of info box
    :param rec_h: height if info box
    :return: a tuple of x and y offset
    """
    ofst_x = ofst_y = round(max(img.shape) * 0.01)

    if pos == "tl":
        pass
    elif pos == "tr":
        ofst_x = img_shape[1] - rec_w - ofst_x
    elif pos == "bl":
        ofst_y = img_shape[0] - rec_h - ofst_y
    elif pos == "br":
        ofst_x = img_shape[1] - rec_w - ofst_x
        ofst_y = img_shape[0] - rec_h - ofst_y

    return ofst_x, ofst_y


if __name__ == "__main__":

    images_dir = os.path.abspath("D:\\BigVision\\library-work\\examples\\resources\\images")

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
