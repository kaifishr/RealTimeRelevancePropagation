"""Class for recording real-time relevance scores.
"""
import cv2


class VideoRecorder(object):

    def __init__(self, conf):
        save_dir = conf["paths"]["save_dir"]
        display_size = conf["playback"]["display_size"]
        fps = 20

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        filename = save_dir + "real_time_lrp.avi"
        self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (display_size, display_size))

    def record(self, image):
        self.video_writer.write(image)

    def release(self):
        self.video_writer.release()
