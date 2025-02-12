from .modelinterface import ModelInterface, Result
from ultralytics import YOLO


class PlainYoloDetector(ModelInterface):
    wrapper: YOLO

    def __init__(self, model_path: str, **args):
        self.wrapper = YOLO(model_path, **args)

    def detect(self, image, raw):
        results = self.wrapper.predict(raw, stream=True)

        for result in results:
            boxes: list[list] = result.boxes.xyxy.tolist()
            names: list[int] = result.boxes.cls.tolist()
            confidences: list[float] = result.boxes.conf.tolist()

            for i in range(len(names)):
                box = boxes[i]
                name = self.wrapper.names[int(names[i])]
                confidence = confidences[i]

                yield Result(
                    name=name,
                    confidence=confidence,
                    point1=(round(box[0]), round(box[1])),
                    point2=(round(box[2]), round(box[3])),
                )
