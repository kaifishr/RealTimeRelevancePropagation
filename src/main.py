"""Real time relevance propagation.

This program uses the frames of a webcam, feeds them into a pre-trained VGG16 or VGG19 network
and performs layer-wise relevance propagation in real time. If desired, a video can be recorded.

For better experience use a fast GPU. ESC stops the program.
"""
import cv2
import time
import yaml

from utils import post_processing
from webcam import Webcam
from video_recorder import VideoRecorder
from relevance_propagation import RelevancePropagation


def real_time_lrp(conf):
    """Method to display feature relevance scores in real time.

    Args:
        conf: Dictionary consisting of configuration parameters.
    """
    record_video = conf["playback"]["record_video"]

    webcam = Webcam()
    lrp = RelevancePropagation(conf)

    if record_video:
        recorder = VideoRecorder(conf)

    while True:
        t0 = time.time()

        frame = webcam.get_frame()
        heatmap = lrp.run(frame)
        heatmap = post_processing(frame, heatmap, conf)
        cv2.imshow("LRP", heatmap)

        if record_video:
            recorder.record(heatmap)

        t1 = time.time()
        fps = 1.0/(t1 - t0)
        print("{:.1f} FPS".format(fps))

        if cv2.waitKey(1) % 256 == 27:
            print("Escape pressed.")
            break

    if record_video:
        recorder.release()

    webcam.turn_off()
    cv2.destroyAllWindows()


def main():
    conf = yaml.safe_load(open("config.yml"))
    real_time_lrp(conf)


if __name__ == '__main__':
    main()
