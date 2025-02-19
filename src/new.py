from .modelinterface import ModelInterface, Result

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter

from ultralytics.utils import ops
from ultralytics.engine.results import Boxes

import yaml
import zipfile

from numpy import expand_dims


class FastDetector(ModelInterface):
    names: list[str]
    model: Interpreter

    def __init__(self, model_path: str, **args):
        with open(model_path, "rb") as file:
            input = None
            with zipfile.ZipFile(file, "r") as zf:
                input = zf.read("temp_meta.txt")

            output = yaml.safe_load(input.decode("utf-8"))

            self.names = output["names"]

            # Return to the beginning of the file
            file.seek(0)

            self.model = Interpreter(model_content=file.read(), num_threads=4)
            self.model.allocate_tensors()

            # Get input and output tensors.
            self.input_details = self.model.get_input_details()
            self.output_details = self.model.get_output_details()

            input_shape = self.input_details[0]["shape"]

            self.height = int(input_shape[1])
            self.width = int(input_shape[2])

    def detect(self, image, raw):
        image = expand_dims(image, axis=0)
        self.model.set_tensor(self.input_details[0]["index"], image)

        self.model.invoke()

        output_tensor = self.model.get_tensor(self.output_details[0]["index"])

        # TODO: Maybe rewrite this for ourselves
        output = ops.non_max_suppression(
            output_tensor,
            conf_thres=0.333,
            iou_thres=0.7,
            nc=len(self.names),
        )[0]

        # See ultralytics/models/yolo/detect/predict.py#L59
        boxes = Boxes(output[:, :6], (1 / self.height, 1 / self.width))

        bounds: list[list] = boxes.xyxyn.tolist()
        ournames: list[int] = boxes.cls.tolist()
        confidences: list[float] = boxes.conf.tolist()

        for i in range(len(ournames)):
            yield Result(
                name=self.names[ournames[i]],
                confidence=confidences[i],
                point1=(round(bounds[i][0]), round(bounds[i][1])),
                point2=(round(bounds[i][2]), round(bounds[i][3])),
            )
