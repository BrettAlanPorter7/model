from .cameras import CameraInterface
from .models import ModelInterface
import cv2
import time


def object_detection_loop(camera: CameraInterface, model: ModelInterface):
    # Variables to calculate FPS
    counter, fps = 0, 0.0
    start_time = time.time()
    fps_avg_frame_count = 10

    while True:
        image = run_object_detection(camera, model)

        # Calculate the FPS
        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()

        # Show the FPS
        fps_text = "FPS = {:.1f}".format(fps)
        text_location = (24, 24)
        cv2.putText(
            image,
            fps_text,
            text_location,
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 0, 255),
            2,
        )

        cv2.imshow("Camera", image)

        counter += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def run_object_detection(camera: CameraInterface, model: ModelInterface):
    array, image = camera.capture()

    results = model.detect(array, image)

    for result in results:
        # Detection box
        cv2.rectangle(
            img=image,
            pt1=result.point1,
            pt2=result.point2,
            color=(10, 255, 0),
            thickness=2,
        )

        # Detection box label
        label = f"{result.name}: {int(result.confidence * 100)}%"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        label_ymin = max(result.point1[1], label_size[1] + 10)
        cv2.rectangle(
            image,
            (int(result.point1[0]), label_ymin - label_size[1] - 10),
            (int(result.point1[0]) + label_size[0], label_ymin + 2),
            (255, 255, 255),
            cv2.FILLED,
        )

        cv2.putText(
            image,
            label,
            (int(result.point1[0]), label_ymin - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

    return image
