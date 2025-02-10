from tflite_runtime.interpreter import Interpreter
from tflite_support import metadata

from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import ops
from ultralytics.engine.results import Results, Boxes
import numpy as np
import cv2
import time
import yaml

displayer = metadata.MetadataDisplayer.with_model_file("models/best_float32.tflite")
tags = []
for char in displayer.get_associated_file_buffer("temp_meta.txt"):
    tags.append(chr(char))
other_data = "".join(tags)
metadata = yaml.safe_load(other_data)

names = metadata["names"]

# Start capturing video input from the camera
cap = cv2.VideoCapture(0)

actual_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

height = min(640, actual_frame_height)
width = min(640, actual_frame_width)

model = Interpreter(model_path="models/best_float32.tflite", num_threads=8)
model.allocate_tensors()

# Get input and output tensors.
input_details = model.get_input_details()
output_details = model.get_output_details()

# Variables to calculate FPS
counter, fps = 0, 0
start_time = time.time()

input_shape = input_details[0]["shape"]

height = int(input_shape[1])
width = int(input_shape[2])

processor = DetectionPredictor(overrides={"conf": 0.25})

while True:
    image = cap.read()[1]

    counter += 1
    # image = cv2.flip(image, 1)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    # input_tensor = vision.TensorImage.create_from_array(rgb_image)

    input = rgb_image.astype(np.float32)
    input /= 255.0
    input = np.expand_dims(input, axis=0)

    model.set_tensor(input_details[0]["index"], input)

    model.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_tensor = model.get_tensor(output_details[0]["index"])

    result = ops.non_max_suppression(
        output_tensor,
        conf_thres=0.333,
        iou_thres=0.7,
        nc=len(names),
    )[0]

    # See ultralytics/models/yolo/detect/predict.py#L59
    boxes = Boxes(result[:, :6], (1 / height, 1 / width))

    bounds: list[list] = boxes.xyxyn.tolist()
    ournames: list[int] = boxes.cls.tolist()
    confidences: list[float] = boxes.conf.tolist()

    for i in range(len(ournames)):
        box = bounds[i]
        ourname = names[int(ournames[i])]
        confidence = confidences[i]

        point1 = (round(box[0]), round(box[1]))
        point2 = (round(box[2]), round(box[3]))
        cv2.rectangle(
            img=image,
            pt1=point1,
            pt2=point2,
            color=(10, 255, 0),
            thickness=2,
        )

        label = f"{ourname}: {int(confidence * 100)}%"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_ymin = max(int(box[1]), label_size[1] + 10)
        cv2.rectangle(
            image,
            (int(box[0]), label_ymin - label_size[1] - 10),
            (int(box[0]) + label_size[0], label_ymin + 2),
            (255, 255, 255),
            cv2.FILLED,
        )
        cv2.putText(
            image,
            label,
            (int(box[0]), label_ymin - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

    # Display the image with bounding boxes
    cv2.imshow("image", image)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
