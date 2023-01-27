import os

import cv2

from dvlabs.utils import create_grid, check_and_create_dir, get_vid_writer, get_batches


class Compare:
    def __init__(self, analysis_obj1, analysis_obj2):

        assert analysis_obj1.gt_annos == analysis_obj2.gt_annos, "The analysis objects belong to different datasets."

        self.anal_obj1 = analysis_obj1
        self.anal_obj2 = analysis_obj2

    def view(self, resolution=(1280, 720), save_dir=None, maintain_ratio=True, filter_classes=[],
             iou_thres=1, show_labels=True, show_conf=True):

        print("Creating View...")

        if resolution[0] < resolution[1]:
            grid_size = [1, 2]
            indv_res = (resolution[0], round(resolution[1]/2))
        else:
            grid_size = [2, 1]
            indv_res = (round(resolution[0]/2), resolution[1])

        # Initialize video writer
        vid_writer = None
        if save_dir is not None:
            # Create directory if not present
            check_and_create_dir(save_dir)
            vid_name = "comparison_output"
            vid_writer = get_vid_writer(os.path.join(save_dir, vid_name), 1,
                                        (indv_res[0] * grid_size[0], indv_res[1] * grid_size[1]))

        image_ids, filtered_gt_annos1, filtered_pred_annos1, _ \
            = self.anal_obj1.filter_anno(filter_classes, iou_thres, view_mistakes=False)

        image_ids, filtered_gt_annos2, filtered_pred_annos2, _ \
            = self.anal_obj2.filter_anno(filter_classes, iou_thres, view_mistakes=False)

        batches = get_batches(image_ids, 1)
        batch_idx = 0

        while True:
            frame1 = self.anal_obj1.process_grid_batch(batches[batch_idx], filtered_gt_annos1, filtered_pred_annos1,
                                                       (1, 1), (indv_res[1], indv_res[0]), maintain_ratio, show_labels,
                                                       show_conf)

            frame2 = self.anal_obj2.process_grid_batch(batches[batch_idx], filtered_gt_annos2, filtered_pred_annos2,
                                                       (1, 1), (indv_res[1], indv_res[0]), maintain_ratio, show_labels,
                                                       show_conf)

            combined_grid = create_grid([frame1, frame2], grid_size, (indv_res[1], indv_res[0]), maintain_ratio)

            # Write grid frame to video or show in window
            if vid_writer is not None:
                vid_writer.write(combined_grid)
                if batch_idx == (len(batches) - 1):  # If last frame
                    # Release video writer
                    vid_writer.release()
                    break
                batch_idx += 1
            else:
                cv2.imshow('grid', combined_grid)
                key = cv2.waitKey(0)
                if key == 27 or key == ord('q'):  # Esc or 'Q' to exit the grid view
                    break
                elif key == 97:  # 'A' for previous frame
                    if batch_idx > 0:
                        batch_idx -= 1
                elif key == 100:  # 'D' for next frame
                    if batch_idx < (len(batches) - 1):
                        batch_idx += 1
