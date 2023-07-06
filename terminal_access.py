import json
import cv2
import time
import socket
import datetime
import sys
import logging
from pathlib import Path
from ultralytics import YOLO


def main():
    model = YOLO('surveillance.pt')
    i = 0

    # Set video source (0 for default camera, or path to a video file)
    video_path = 0
    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Initialize the time variable
    prev_inference_time = 0

    # Loop through video frames
    while cap.isOpened():
        i += 1
        # Read a frame
        success, frame = cap.read()
        # Resize the frame
        frame = resize_frame(frame, width=1280, height=720)
        # Break the loop if the end of the video is reached
        if not success:
            break

        # Get the current time
        current_time = time.time()

        # Check if at least 0.5 seconds have passed since the last inference
        if current_time - prev_inference_time >= 0.1:
            # Run YOLOv8 inference
            results = model(frame, verbose=False)
            p = Path('dummy_path')
            im = frame
            im0 = frame.copy()
            label = model.predictor.custom_results(0, results, (p, im, im0))
            # print(f"****{label}*****")

            # json response
            time_stamp = datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S")
            if "Civilian" in label:
                response = {"command": "security", "action": "trigger", "tags":f"event={label}", "Date-Time": time_stamp, }
            else:
                response = {"command": "security", "action": "", "tags":f"event={label}"}
            # Send the JSON response
            send_json_response(ip, port, response)

            #Json File testing
            # response_string = json.dumps(response)
            # with open("test_{i}.json", "w") as f:
            #     f.write(response_string)

            # Visualize results on the frame
            for result in results:
                annotated_frame = result.plot()
                # Display the annotated frame using cv2.imshow
                cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Update the previous inference time
            prev_inference_time = current_time
        else:
            # Display the unannotated frame
            cv2.imshow("YOLOv8 Inference", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


def resize_frame(frame, width, height):
    # Resize the frame to the specified dimensions
    return cv2.resize(frame, (width, height))

def send_json_response(ip, port, response):
    """
    Sends a JSON response over the given IP and port.

    Args:
        ip (str): The IP address of the server.
        port (int): The port of the server.
        response (dict): The JSON response.
    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))

    # Convert the JSON response to a string.
    response_string = json.dumps(response)

    # Send the JSON response to the server.
    sock.sendall(bytes(response_string, "utf-8"))

    # Close the socket.
    sock.close()


if __name__ == "__main__":
    ip = "10.0.1.20"
    port = 2101
    main()