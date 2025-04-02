from .cameras import CameraInterface
from .models import ModelInterface
import cv2
import time
from mjpeg_streamer import Stream, MjpegServer
from pymavlink import mavutil


def object_detection_loop(camera: CameraInterface, model: ModelInterface):
    # Variables to calculate FPS
    counter, fps = 0, 0.0
    start_time = time.time()
    fps_avg_frame_count = 10

    stream = Stream("picam", size=(640, 640), fps=5)
    server = MjpegServer("0.0.0.0", 8080)
    server.add_stream(stream)
    server.start()

    print("Connecting to drone...")
    master = mavutil.mavlink_connection('udp:127.0.0.1:14550')  #adjust for actual drone connection
    master.wait_heartbeat()  #make sure it is connected
    print("✅ Drone connected!")

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

        # cv2.imshow("Camera", image)
        stream.set_frame(image)

        counter += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def run_object_detection(camera: CameraInterface, model: ModelInterface):
    array, image = capture() #pulls array, image from capture
	results = model.detect(array, image) #potentially need to fix this so we get the person detection, maybe results = (array)?

	no_person_detected = all(res.name != "person" for res in results)

	msg = master.recv_match(type='HEARTBEAT', blocking=True) #pulls heartbeat message
	mode = mavutil.mode_string_v10(msg) # pulls flight mode from heartbeat message
	print(f"Current flight mode: {mode}")

	if no_person_detected: 
		if mode != 'LOITER':
			#print("✅ No person detected") removed so this is not constantly printed, is continue the correct thing to put next?
			continue

		else: #meaning the drone is currently in loiter mode
			print("✅ No person detected")
			#change mode to auto because no person detected
			master.mav.command_long_send(
				master.target_system, master.target_component,
				mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
				0,  # Base mode (ignored)
				mavutil.mavlink.enums['MAV_MODE_AUTO_DISARMED'].value,
				0, 0, 0, 0, 0  # Unused parameters
			)

	else: # person is detected
		if mode != 'LOITER':
			print("🚨 Human detected! Halting drone for 5 seconds... 🚨")
			#change mode to loiter because person detected
			master.mav.command_long_send(
				master.target_system, master.target_component,
				mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
				0,  # Base mode (ignored)
				mavutil.mavlink.enums['MAV_MODE_LOITER'].value,
				0, 0, 0, 0, 0  # Unused parameters
			)
			time.sleep(5) #wait 5 seconds

		else: #drone already in loiter mode
			print("🚨 Human detected! Halting drone for 5 seconds... 🚨")
			time.sleep(5) #wait 5 seconds

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
