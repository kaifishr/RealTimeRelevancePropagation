"""Python class to capture webcam frames.

  Typical usage example:

  webcam = Webcam()
  frame = webcam.get_frame()
  webcam.turn_off()
"""
import cv2


class Webcam(object):
    """Returns central crop of webcam frame as Numpy array.
    """

    def __init__(self, camera=0):
        self.cam = cv2.VideoCapture(camera)

        if not self.cam.isOpened():
            self.cam.open()

        width = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.min_length = min(width, height)

    def __capture(self):
        return_value, img = self.cam.read()
        if not return_value:
            print("Error: No frame captured.")
        img = cv2.flip(img, 1)
        return img

    def get_frame(self, target_size=224):
        """Returns webcam frame.

        Returns: Numpy array of specified size.
        """
        img = self.__capture()
        img = self.square_center_crop(img=img, dim=self.min_length)
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        return img

    def turn_off(self):
        """Turns off webcam.
        """
        self.cam.release()

    @staticmethod
    def square_center_crop(img, dim):
        """Gets largest central crop of webcam frame to avoid distortions.

        Args:
            img: Numpy array.
            dim: Integer representing webcam's resolution along y-axis.

        Returns:
            Numpy array of central crop.

        """
        width, height = img.shape[1], img.shape[0]
        crop_width = dim if dim < img.shape[1] else img.shape[1]
        crop_height = dim if dim < img.shape[0] else img.shape[0]
        x, y = int(0.5 * width), int(0.5 * height)
        dx, dy = int(0.5 * crop_width), int(0.5 * crop_height)
        return img[y - dy:y + dy, x - dx:x + dx]
