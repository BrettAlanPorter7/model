import ultralytics
import cv2
import time
import sys
from picamera2 import Picamera2
from libcamera import Transform

# model = ultralytics.YOLO("./best.pt")
model = ultralytics.YOLO("./best_float32.tflite")

# Variables to calculate FPS
counter, fps = 0, 0
start_time = time.time()

# Start capturing video input from the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# Visualization parameters
row_size = 20  # pixels
left_margin = 24  # pixels
text_color = (0, 0, 255)  # red
font_size = 1
font_thickness = 1
fps_avg_frame_count = 10



camera = Picamera2()
configuration = camera.create_preview_configuration(main={
    'size': (640, 640),
    'format': 'BGR888'
}, transform=Transform(hflip=1))

camera.configure(configuration)

camera.start()

while True:
    image = camera.capture_array()

    counter += 1
    image = cv2.flip(image, 1)

    results = model.predict(image, stream=True, verbose=True)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for result in results:
        boxes: list[list] = result.boxes.xyxy.tolist()
        names: list[int] = result.boxes.cls.tolist()
        confidences: list[float] = result.boxes.conf.tolist()

        for i in range(len(names)):
            box = boxes[i]
            name = model.names[int(names[i])]
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

            label = f"{name}: {int(confidence * 100)}%"
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

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
        end_time = time.time()
        fps = fps_avg_frame_count / (end_time - start_time)
        start_time = time.time()

    # Show the FPS
    fps_text = "FPS = {:.1f}".format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(
        image,
        fps_text,
        text_location,
        cv2.FONT_HERSHEY_PLAIN,
        font_size,
        text_color,
        font_thickness,
    )

    cv2.imshow("object_detector", image)
    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
