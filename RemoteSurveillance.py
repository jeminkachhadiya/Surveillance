import argparse
import json
import os
import threading
import cv2
import time
import socket
import datetime
import supervision as sv
from pathlib import Path
from ultralytics import YOLO
import numpy as np


def getPicture(input):
    ip, port, rtsp, inference_path, img_inference, video_inference, results_visualization, wait_interval, \
        conf_threshold, device = opt.ip, opt.port, opt.rtsp, opt.inference_path, opt.img_inference, \
        opt.video_inference, opt.results_visualization, opt.wait_interval, opt.conf_threshold, opt.device
    model = YOLO('surveillance.pt')

    # Create a VideoCapture object
    cap = cv2.VideoCapture(input)

    # to save the video
    writer = cv2.VideoWriter(os.path.join(inference_path,'surveillance.mp4'), 
                    cv2.VideoWriter_fourcc(*'mp4v'), 
                    int(cap.get(5)), (int(cap.get(3)), int(cap.get(4))))
    
    a=cap.get(cv2.CAP_PROP_BUFFERSIZE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,3)
    start_frame_number = 20
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)

    # Initialize the variables
    prev_inference_time = 0
    response = ''

    # customize the bounding box
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    # Loop through video frames
    while cap.isOpened():
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
        if current_time - prev_inference_time >= wait_interval:
            # Run YOLOv8 inference
            results = model(frame, verbose=False, agnostic_nms=True, conf=conf_threshold, device=device)
            result = results[0]
            detections = sv.Detections.from_yolov8(result)
            # detections = list_of_results(results[0])
            
            # time stamp for json response
            time_stamp = datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S")
            time_stamp_name = datetime.datetime.now().strftime("%m-%d-%Y_%Hh%Mm%Ss")

            try:
                labels = [
                f"{model.model.names[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _ 
                in detections
                ]
                # print(detections)
                print(f"{labels}\n{time_stamp}")

                annotated_frame = box_annotator.annotate(
                scene=frame, 
                detections=detections, 
                labels=labels
            )
            except Exception as e:
                print(e)
                continue
            
            p = Path('dummy_path')
            im = frame
            im0 = frame.copy()
            label = model.predictor.custom_results(0, results, (p, im, im0))

            # Label format for json response
            civilians = []
            soldiers = []
            for item in labels:
                if item.startswith('Civilian'):
                    civilians.append(float(item.split()[1]))
                elif item.startswith('Soldier'):
                    soldiers.append(float(item.split()[1]))

            if "Civilian" in label and "Soldier" in label:
                response = {"command": "security", "action": "trigger", 
                            "tags":f"event=civilian&civilians={civilians}&soldiers={soldiers}",
                            "Date-Time": time_stamp}
                if video_inference:
                    writer.write(annotated_frame)
                if img_inference:
                    cv2.imwrite(f'civilian_soldier_{time_stamp_name}.jpg', annotated_frame)
            elif "Civilian" in label:
                response = {"command": "security", "action": "trigger", 
                            "tags":f"event=civilian&civilians={civilians}",
                            "Date-Time": time_stamp}
                if video_inference:
                    writer.write(annotated_frame)
                if img_inference:
                    cv2.imwrite(os.path.join(inference_path, f'civilian_{time_stamp_name}.jpg'), annotated_frame)
                # cv2.imshow("Inference", frame)
            elif "Soldier" in label:
                response = {"command": "security", "action": "trigger", 
                            "tags":f"event=civilian&soldiers={soldiers}",
                            "Date-Time": time_stamp}
                
            # Send the JSON response
            """if response:
                send_json_response(ip, port, response)"""

            # Json File testing
            if response:
                response_string = json.dumps(response)
                with open("test.json", "a") as f:
                    f.write(response_string+"\n")

            # Visualize results on the frame
            if results_visualization:
                cv2.imshow("Inference", annotated_frame)

            # Update the previous inference time
            prev_inference_time = current_time
        else:
            # Display the unannotated frame
            if results_visualization:
                cv2.imshow("Inference", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close the display window
    cap.release()
    # writer.release()
    cv2.destroyAllWindows()

def list_of_results(results):
    global confidence
    global class_id
    return results.boxes.conf.cpu().numpy(), results.boxes.cls.cpu().numpy().astype(int)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default="10.0.1.20", help='ip address for json response')
    parser.add_argument('--port', type=int, default=2101, help='source in url or video file path')
    parser.add_argument('--rtsp', type=str, default=0, help='rtsp address or 0 for webcam input')
    parser.add_argument('--inference-path', type=str, default="./", help='path location to save all the inferences')
    parser.add_argument('--img-inference', type=bool, default=False, help='Save image inferences on given inference path')
    parser.add_argument('--video-inference', type=bool, default=False, help='Save video on inferences path')
    parser.add_argument('--results-visualization', type=bool, default=False, help='display processed frames with bounding box')
    parser.add_argument('--wait-interval', type=float, default=0.1, help="buffer period for frames")
    parser.add_argument('--conf-threshold', type=float, default=0.35, help="confidence threshold for object detection")
    parser.add_argument('--device', default=0, help="0/1/2/3 for GPU, 'cpu' for CPU")
    opt = parser.parse_args()
    t1 = threading.Thread(target=getPicture, args=(opt.rtsp,))
    # t2 = threading.Thread(target=getPicture, args=(rtsp1,))
    t1.start()
    # t2.start()
