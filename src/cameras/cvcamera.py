from .camerainterface import CameraInterface
from numpy import float32
import cv2


class Cv2Camera(CameraInterface):
    camera: cv2.VideoCapture

    def __init__(self, height=640, width=640, **kwargs):
        super().__init__(height, width, **kwargs)

        self.camera = cv2.VideoCapture(kwargs.get("input", 0))
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)

        # Test it once to see if it works.
        self.capture()

    def capture(self):
        success, frame = self.camera.read()

        if not success:
            raise RuntimeError(
                "ERROR: Unable to read from webcam. Please verify your webcam settings."
            )

        frame = cv2.flip(frame, 1)

        frame = cv2.resize(frame, (self.height, self.width))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        float_array = rgb.astype(float32)
        float_array /= 255.0

        return float_array, frame

    def stop_camera(self):
        super().stop_camera()
        self.camera.release()
