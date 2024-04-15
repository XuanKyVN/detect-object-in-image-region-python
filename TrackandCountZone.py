import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import time
# Load the YOLOv8 model
model = YOLO('C:/Users/Admin/PythonLession/YoloModel/yolov8s.pt')

# Open the video file
video_path = "C:/Users/Admin/PythonLession/pic/people1.mp4"
cap = cv2.VideoCapture(video_path)


pts = np.array([[0,100],[800,100],[800,500],[0,500],[0,100]], np.int32)
pts = pts.reshape((-1,1,2))
crossed_objects = {}
track_history = defaultdict(lambda: [])


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    frame=cv2.resize(frame,(1200,750))

    cv2.polylines(frame, [pts], True, (255,0,0), 2)

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, classes = [0])
        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        # Visualize the results on the frame
        #annotated_frame = results[0].plot()
        annotated_frame =frame
        countperson =0
        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)
            # Drawn the center point for object
            dist = cv2.pointPolygonTest(pts, (int(x), int(y)), False)
            if dist ==1:
                cv2.circle(annotated_frame, (int(x), int(y)), 6, (0, 255, 0), 3)
                if track_id not in crossed_objects:
                    crossed_objects[track_id] = True
                cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),(0, 255, 0), 2)
                countperson +=1 # Count persons by Polygon
            else:
                cv2.circle(annotated_frame, (int(x), int(y)), 6, (0, 0, 255), 2)


        # Counting CAR passing the zone
        count_text = f"People_COUNT: {len(crossed_objects)}"
        cv2.putText(annotated_frame, count_text, (121, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(annotated_frame, "CountRegion: "+str(countperson), (121, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        # Display the annotattracked frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # DELAY THE FRAME FOR CHECKING COUNTER
        #time.sleep(0.1)
        print(crossed_objects)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()