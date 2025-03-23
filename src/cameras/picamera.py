from numpy import float32
from .camerainterface import CameraInterface
from picamera2 import Picamera2

try:
    from libcamera import Transform
except ImportError:
    raise RuntimeError(
        "ERROR: Please install the libcamera library to use the PiCamera."
    )


class PiCamera(CameraInterface):

    camera: Picamera2

    def __init__(self, height=640, width=640, **kwargs):
        super().__init__(height, width, **kwargs)

        self.camera = Picamera2()
        # BGR888 is actually RGB, yes, I know, it's confusing.
        # RGB is what the model can take as an input.
        configuration = self.camera.create_preview_configuration(
            main={"size": (height, width), "format": "BGR888"},
            transform=Transform(hflip=1),
        )

        self.camera.configure(configuration)

        self.camera.start()

    def capture(self):
        value = self.camera.capture_array()

        float_array = value.astype(float32)
        float_array /= 255.0

        return float_array, value

    def stop_camera(self):
        super().stop_camera()
        self.camera.close()
