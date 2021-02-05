"""Helper method for real-time relevance propagation.
"""
import cv2
import numpy as np


def post_processing(frame, heatmap, conf):
    alpha = conf["playback"]["alpha"]
    display_size = conf["playback"]["display_size"]

    heatmap = (255.0 * heatmap).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    img = (alpha * heatmap + (1-alpha) * frame).astype(np.uint8)
    img = cv2.resize(img, (display_size, display_size), interpolation=cv2.INTER_NEAREST)

    return img.astype(np.uint8)
