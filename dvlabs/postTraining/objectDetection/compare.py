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

        grid_frames1 = self.get_frames(self.anal_obj1, filter_classes, iou_thres, indv_res, maintain_ratio,
                                       show_labels, show_conf)
        grid_frames2 = self.get_frames(self.anal_obj1, filter_classes, iou_thres, indv_res, maintain_ratio,
                                       show_labels, show_conf)

        frame_idx = 0

        while True:
            frame1 = grid_frames1[frame_idx]
            frame2 = grid_frames2[frame_idx]

            combined_grid = create_grid([frame1, frame2], grid_size, (indv_res[1], indv_res[0]), maintain_ratio)

            # Write grid frame to video or show in window
            if vid_writer is not None:
                vid_writer.write(combined_grid)
                if frame_idx == (len(grid_frames1) - 1):  # If last frame
                    # Release video writer
                    vid_writer.release()
                    break
                frame_idx += 1
            else:
                cv2.imshow('grid', combined_grid)
                key = cv2.waitKey(0)
                if key == 27 or key == ord('q'):  # Esc or 'Q' to exit the grid view
                    break
                elif key == 97:  # 'A' for previous frame
                    if frame_idx > 0:
                        frame_idx -= 1
                elif key == 100:  # 'D' for next frame
                    if frame_idx < (len(grid_frames1) - 1):
                        frame_idx += 1

    def get_frames(self, anal_obj, filter_classes, iou_thres, indv_res, maintain_ratio, show_labels, show_conf):
        filtered_image_ids, filtered_gt_annos, filtered_pred_annos, combined_mistakes_anno \
            = anal_obj.filter_anno(filter_classes, iou_thres, view_mistakes=False)

        batches = get_batches(filtered_image_ids, 1)

        grid_frames = []

        for batch in batches:
            grid = anal_obj.process_grid_batch(batch, filtered_gt_annos, filtered_pred_annos, (1, 1),
                                               (indv_res[1], indv_res[0]), maintain_ratio, show_labels, show_conf)
            grid_frames.append(grid)

        return grid_frames
