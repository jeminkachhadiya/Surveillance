from ultralytics.yolo.engine.model import YOLO
import json
import cv2
import os

model = YOLO("surveillance.pt")
results = model.predict(0, save=True, show=True)
for r in results:
    print(r)
# results.json()
# dump_to_json(f"{os.path.basename(image_path)}.json", json_results)

# def classify_video_stream(video_path):
#     model = YOLO("surveillance.pt")
#     metrics_t = model.predict(0, save=True, show=True)
#     print(metrics_t)

#     # Create a video capture object.
#     capture = cv2.VideoCapture(video_path)

#     # Create a JSON object to store the classification response.
#     response = {}
#     response["classes"] = []

#     # Iterate over the frames in the video.
#     while True:
#         # Capture the next frame.
#         ret, frame = capture.read()

#         # Detect objects in the frame.
#         detections = model.detect(frame)

#         # Iterate over the detections and add them to the JSON object.
#         for detection in detections:
#             class_name = detection["name"]
#             confidence = detection["confidence"]

#             # Add the class name and confidence to the JSON object.
#             response["classes"].append({"class_name": class_name, "confidence": confidence})

#         # Print the JSON object.
#         print(json.dumps(response, indent=4))

#         # Break the loop if the user presses ESC.
#         if cv2.waitKey(1) == 27:
#             break

#     # Close the video capture object.
#     capture.release()


# if __name__ == "__main__":
#     classify_video_stream(0)