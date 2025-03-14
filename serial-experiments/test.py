import cv2
import numpy as np

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

position = 0
while True:
    success, frame = camera.read()
    if not success:
        break

    frame = cv2.resize(frame, (64, 64))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
