import json
import threading
import cv2
import time
import socket
import datetime
import supervision as sv
from pathlib import Path
from ultralytics import YOLO


def getPicture(input):
    model = YOLO('surveillance.pt')

    # Create a VideoCapture object
    cap = cv2.VideoCapture(input)

    # to save the video
    # writer= cv2.VideoWriter('suspicious_civilian.mp4', 
    #                         cv2.VideoWriter_fourcc(*'DIVX'), 
    #                         7, 
    #                         (1280, 720))
    
    a=cap.get(cv2.CAP_PROP_BUFFERSIZE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,3)
    start_frame_number = 20
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)

    # Initialize the time variable
    prev_inference_time = 0

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
        if current_time - prev_inference_time >= 0.1:
            # Run YOLOv8 inference
            results = model(frame, verbose=False, agnostic_nms=True, conf=0.35)
            detections = sv.Detections.from_yolov8(results[0])
            labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _ in detections
            ]
            # print(labels)

            annotated_frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
            )
            
            p = Path('dummy_path')
            im = frame
            im0 = frame.copy()
            label = model.predictor.custom_results(0, results, (p, im, im0))

            # json response
            time_stamp = datetime.datetime.now().strftime("%m/%d/%Y %H:%M:%S")
            time_stamp_name = datetime.datetime.now().strftime("%m-%d-%Y_%Hh%Mm%Ss")
            if "Civilian" in label:
                response = {"command": "security", "action": "trigger", "tags":f"event={labels}", "Date-Time": time_stamp, }
                # writer.write(annotated_frame)
                cv2.imwrite(f'civilian_{time_stamp_name}.jpg', annotated_frame)
                # cv2.imshow("Inference", frame)
            else:
                response = {"command": "security", "action": "", "tags":f"event={labels}", "Date-Time": time_stamp,}
            # Send the JSON response
            # send_json_response(ip, port, response)

            # Json File testing
            response_string = json.dumps(response)
            with open("test.json", "a") as f:
                f.write(response_string+"\n")

            # Visualize results on the frame
            cv2.imshow("Inference", annotated_frame)

            # Update the previous inference time
            prev_inference_time = current_time
        else:
            # Display the unannotated frame
            cv2.imshow("Inference", frame)

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
    rtsp = ""
    t1 = threading.Thread(target=getPicture, args=(0,))
    # t2 = threading.Thread(target=getPicture, args=(rtsp1,))

    t1.start()
    # t2.start()