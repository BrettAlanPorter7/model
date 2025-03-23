from abc import abstractmethod, ABCMeta
from numpy import ndarray
from cv2.typing import MatLike


class CameraInterface(metaclass=ABCMeta):
    hasStopped = False
    height: int
    width: int

    @abstractmethod
    def __init__(self, height=640, width=640, **kwargs):
        self.height = height
        self.width = width

    @abstractmethod
    def capture(self) -> tuple[ndarray, MatLike]:
        """
        Should return the processed RGB array for the model's use, as well as the original capture for previewing.
        Both should be the proper dimensions.
        """
        pass

    @abstractmethod
    def stop_camera(self):
        self.hasStopped = True
        pass

    # For `with` statements
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_camera()

    # If it gets garbage collected before stopping
    def __del__(self):
        if not self.hasStopped:
            self.stop_camera()
            raise RuntimeWarning(
                "Camera not stopped, remember to call camera.stop_camera() after you're done! Stopping now..."
            )
