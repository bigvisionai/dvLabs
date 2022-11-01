import os
import cv2


def display(img, fps=24.99, algo_name="SE "):

    fps_str = "FPS : " + str(round(fps, 2))

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 0)

    thickness = 2
    fontScale = .8

    name_thck = 2
    name_scale = 1

    # const = 600
    # fontScale = min(img.shape[0], img.shape[1]) / (const / fontScale)
    # name_scale = min(img.shape[0], img.shape[1]) / (const / name_scale)

    c = 1/22
    fontScale = img.shape[1] * .03 * c
    name_scale = img.shape[1] * .03 * c


    (name_size, name_bline) = cv2.getTextSize(algo_name, font, name_scale, name_thck)
    (fps_size, fps_bline) = cv2.getTextSize("FPS : " + str(999.99), font, fontScale, thickness)
    print(fps_size)
    print(name_size, name_bline)

    rec_w = max(name_size[0], fps_size[0]) + 10
    rec_h = name_size[1] + fps_size[1] + 30

    cv2.rectangle(img, [10, 10, rec_w, rec_h], (0, 255, 255), -1)
    cv2.putText(img, fps_str, (int((rec_w / 2) + 10 - (fps_size[0] / 2)), 35), font, fontScale, color, thickness,
                cv2.LINE_AA)
    cv2.putText(img, algo_name, (int((rec_w / 2) + 10 - (name_size[0] / 2)), 70), font, name_scale, color, name_thck,
                cv2.LINE_AA)

if __name__ == "__main__":

    images_dir = os.path.abspath("D:\\BigVision\\library-work\\examples\\images")

    image_path = os.path.join(images_dir, "000001.jpg")
    # image_path = os.path.join(images_dir, "000001 - copy.jpg")
    # image_path = os.path.join(images_dir, "000001 - Copy (2).jpg")

    img = cv2.imread(image_path)

    display(img)

    cv2.imwrite("test.jpg", img)
    cv2.imshow("test", img)

    cv2.waitKey(0)
