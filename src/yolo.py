from .modelinterface import ModelInterface, Result
from ultralytics import YOLO
from ultralytics.engine.predictor import BasePredictor
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.cfg import get_cfg


class YoloDetector(ModelInterface):
    wrapper: YOLO
    model: DetectionModel
    predictor: BasePredictor

    def __init__(self, model_path: str, **args):
        self.wrapper = YOLO(model_path)

        self.model = self.wrapper.model

        args = {
            **args,
            "verbose": False,
            "save": False,
            "conf": 0.25,
            "batch": 1,
            "task": "detect",
        }

        self.predictor = DetectionPredictor(overrides=args)
        self.predictor.args = get_cfg(self.predictor.args, args)
        self.predictor.setup_model(self.model)

    def detectOld(self, image):
        results = self.wrapper.predict(image, stream=True)

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

    def detect(self, image):
        original_results = self.predictor.stream_inference(source=image)

        for result in original_results:
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
