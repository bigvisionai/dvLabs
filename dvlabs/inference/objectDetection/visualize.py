import cv2

from dvlabs.utils import get_font_scale_n_thickness, denormalize_bbox
from dvlabs.config import lib_annotation_format, yolo_bb_format, label_positions


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
    fps_scale, thickness = get_font_scale_n_thickness(img.shape[:2], scale_factor=0.8)
    line_scale, _ = get_font_scale_n_thickness(img.shape[:2])

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
    :param lines: list of text lines
    :param font: text font
    :param line_scale: text line font scale
    :param thickness: text line font thickness
    :return: tuple of list of text line sizes, maximum width of text and sum of heights of line texts
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
    ofst_x = ofst_y = round(max(img_shape) * 0.01)

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


def display_anno(img, img_anon, class_names, bbox_color=(0, 0, 0), class_colors=None, txt_color=(0, 0, 0),
                 font=cv2.FONT_HERSHEY_SIMPLEX, lbl_pos=label_positions.TL, show_labels=True):

    for obj in img_anon[lib_annotation_format.OBJECTS]:

        if class_colors is not None:
            class_color = class_colors[class_names.index(obj[yolo_bb_format.CLASS])]
        else:
            class_color = bbox_color

        bbox = denormalize_bbox(obj, img_anon[lib_annotation_format.IMG_WIDTH],
                                img_anon[lib_annotation_format.IMG_HEIGHT])

        _, bbox_thickness = get_font_scale_n_thickness((bbox[2]-bbox[0], bbox[3]-bbox[1]), scale_factor=1)
        cv2.rectangle(img, bbox[:2], bbox[2:4], class_color, bbox_thickness)

        if yolo_bb_format.CONF not in obj.keys():
            mark_points(img, bbox, bbox_thickness)

        if show_labels:

            if yolo_bb_format.CONF in (obj.keys()):
                show_label(img, f"{obj[yolo_bb_format.CLASS]} {obj[yolo_bb_format.CONF]}", obj, bbox, font, lbl_pos,
                           class_color, txt_color)
            else:
                show_label(img, f"{obj[yolo_bb_format.CLASS]}", obj, bbox, font, lbl_pos, class_color, txt_color)


def show_label(img, text, obj, bbox, font, lbl_pos, class_color, txt_color):
    lbl_scale, thickness = get_font_scale_n_thickness(img.shape[:2], scale_factor=0.6)

    lbl_box, text_ccord = get_lbl_coord(bbox=bbox, lbl_text=text, font=font,lbl_scale=lbl_scale, thickness=thickness,
                                        lbl_pos=lbl_pos)

    cv2.rectangle(img, lbl_box, class_color, -1)
    cv2.putText(img, text, text_ccord, font, lbl_scale, txt_color, thickness=thickness, lineType=cv2.LINE_AA)


def get_lbl_coord(bbox, lbl_text, font, lbl_scale, thickness, lbl_pos):

    ((lbl_w, lbl_h), lbl_bline) = cv2.getTextSize(lbl_text, font, lbl_scale, thickness)

    if lbl_pos == label_positions.TL:
        lbl_box = [bbox[0], bbox[1]-lbl_h-lbl_bline, lbl_w, lbl_h+lbl_bline]
        text_coord = [bbox[0], bbox[1]-lbl_bline]
    elif lbl_pos == label_positions.TR:
        lbl_box = [bbox[2]-lbl_w, bbox[1]-lbl_h-lbl_bline, lbl_w, lbl_h+lbl_bline]
        text_coord = [bbox[2]-lbl_w, bbox[1]-lbl_bline]
    elif lbl_pos == label_positions.BL:
        lbl_box = [bbox[0], bbox[3], lbl_w, lbl_h+lbl_bline]
        text_coord = [bbox[0], bbox[3]+lbl_h]
    elif lbl_pos == label_positions.BR:
        lbl_box = [bbox[2]-lbl_w, bbox[3], lbl_w, lbl_h+lbl_bline]
        text_coord = [bbox[2]-lbl_w, bbox[3]+lbl_h]

    return lbl_box, text_coord


def mark_points(img, bbox, radius=3, color=0):
    xmin, ymin, xmax, ymax = tuple(bbox)
    cv2.circle(img, (xmin, ymin), radius, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (xmin, ymax), radius, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (xmax, ymin), radius, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (xmax, ymax), radius, color, -1, lineType=cv2.LINE_AA)
