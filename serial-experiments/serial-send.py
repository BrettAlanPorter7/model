from serial import Serial
import cv2
import numpy as np
from pickle import loads, dumps, UnpicklingError
from time import monotonic, sleep

from multiprocessing import Process
from pymavlink import mavutil
from pty import openpty
from os import ttyname

import select

master, slave = openpty()
print(master, slave)
print(ttyname(slave))

laptop = Serial("/dev/tnt0", 9600, rtscts=True)
drone = Serial("/dev/tnt1", 9600, rtscts=True)


def send_data(serial_port: Serial, data):
    serial_port.write(data)
    serial_port.write(b"\0")


def receive_data(serial_port: Serial):
    data = serial_port.read_until(b"\0")
    print(data)


def send_picture(serial_port: Serial):
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    while True:
        if not serial_port.cts:
            continue

        success, frame = camera.read()
        if not success:
            return

        frame = cv2.resize(frame, (64, 64))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
        serial_port.write(dumps(frame, 5))
        serial_port.write(b"ENDOFJPEG\0")


def receive_picture(serial_port: Serial):

    while True:
        try:
            if serial_port.in_waiting < 1:
                serial_port.rts = True
            data = serial_port.read_until(b"ENDOFJPEG\0")
            serial_port.rts = False
            frame = loads(data)

            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
            frame = cv2.resize(frame, (640, 640))

            cv2.imshow("Frame", frame)
            cv2.waitKey(1)

        except UnpicklingError:
            print("Error")


def receive_select(serial_port: Serial):
    rl, wl, el = select.select([serial_port])


SIZE = 32
ROWS = int(SIZE // 0.66666)
SKIP = 1

ROWS = 32


def send_partial_picture(serial_port: Serial):
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    position = 0
    while True:
        success, frame = camera.read()
        if not success:
            return

        frame = cv2.resize(frame, (SIZE, SIZE))
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)

        serial_port.write(frame[position])
        position = (position + SKIP) % ROWS


def receive_partial_picture(serial_port: Serial):
    position = 0
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    timer = monotonic()
    while True:
        if serial_port.in_waiting < 1:
            serial_port.rts = True
        serial_port.rts = False
        data = serial_port.read(64)

        data = np.frombuffer(data, dtype=np.uint8)
        image[position] = data.reshape((32, 3))
        # image[position] = data

        output = image
        # output = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_I420)
        output = cv2.resize(output, (SIZE * 10, SIZE * 10))

        position = (position + SKIP) % ROWS

        if position == 0:
            print(monotonic() - timer)
            cv2.imshow("Frame", output)
            cv2.waitKey(1)
            timer = monotonic()


send = Process(target=send_picture, args=(laptop,))
receive = Process(target=receive_picture, args=(drone,))

laptop.rts = False
# receive.start()
send.start()

receive_picture(drone)

# receive.join()
send.join()
