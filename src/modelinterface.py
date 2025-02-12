from abc import abstractmethod, ABC
from numpy import ndarray
from cv2.typing import MatLike


class Result:
    name: str
    confidence: float
    point1: tuple[int, int]
    point2: tuple[int, int]

    def __init__(
        self,
        name: str,
        confidence: float,
        point1: tuple[int, int],
        point2: tuple[int, int],
    ):
        self.name = name
        self.confidence = confidence
        self.point1 = point1
        self.point2 = point2


class ModelInterface(ABC):

    @abstractmethod
    def __init__(self, model_path: str):
        pass

    @abstractmethod
    def detect(self, image: ndarray, raw: MatLike) -> list[Result]:
        pass
