import cv2
import time
import sys
from src.modelinterface import ModelInterface
from src.yolo import YoloDetector


#model = YoloDetector(model_path="./best.pt")
model: ModelInterface = YoloDetector(model_path="./best_float32.tflite", device="cpu")

# Variables to calculate FPS
counter, fps = 0, 0
start_time = time.time()

# Start capturing video input from the camera
cap = cv2.VideoCapture("test.webm")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# Visualization parameters
row_size = 20  # pixels
left_margin = 24  # pixels
text_color = (0, 0, 255)  # red
font_size = 1
font_thickness = 1
fps_avg_frame_count = 10

# Continuously capture images from the camera and run inference
while cap.isOpened():
    success, image = cap.read()
    if not success:
        sys.exit(
            "ERROR: Unable to read from webcam. Please verify your webcam settings."
        )

    counter += 1
    image = cv2.flip(image, 1)
    image = cv2.resize(image, (640, 640))

    results = model.detect(image)

    for result in results:
        cv2.rectangle(
            img=image,
            pt1=result.point1,
            pt2=result.point2,
            color=(10, 255, 0),
            thickness=2,
        )

        label = f"{result.name}: {int(result.confidence * 100)}%"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        label_ymin = max(result.point1[1], label_size[1] + 10)
        cv2.rectangle(
            image,
            (int(result.point1[1]), label_ymin - label_size[1] - 10),
            (int(result.point1[1]) + label_size[0], label_ymin + 2),
            (255, 255, 255),
            cv2.FILLED,
        )
        cv2.putText(
            image,
            label,
            (int(result.point1[1]), label_ymin - 7),
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
