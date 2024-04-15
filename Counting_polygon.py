from collections import defaultdict
import time
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('C:/Users/Admin/PythonLession/YoloModel/yolov8s.pt')

# Open the video file
#cap = cv2.VideoCapture('C:/Users/Admin/PythonLession/YoloV8/pic/traffic-4ways.mp4')
#video_path="d.mp4"
#cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture('C:/Users/Admin/PythonLession/pic/people1.mp4')

# Store the track history
track_history = defaultdict(lambda: [])
crossed_objects = {}

obcross = 30 # 8 pxel, when object near the line, calculator start couting

# Define Region to count object
#pts =np.array([[282, 232],[918, 196],[1178, 388],[218, 480],[282, 232]],np.int32)
pts = np.array([[94, 312],[1030, 304],[1106, 516],[74, 540],[94, 312]], np.int32)
pts = pts.reshape(-1, 1, 2)

countperson =0

#-----------------------------------------
# Define the line coordinates
START = [sv.Point(300, 650)]
END = [sv.Point(1100, 550)]



#      Function to check distance from a point to a line

from math import sqrt
def minDistance(A, B, E):
    # vector AB
    AB = [None, None];
    AB[0] = B[0] - A[0];
    AB[1] = B[1] - A[1];

    # vector BP
    BE = [None, None];
    BE[0] = E[0] - B[0];
    BE[1] = E[1] - B[1];

    # vector AP
    AE = [None, None];
    AE[0] = E[0] - A[0];
    AE[1] = E[1] - A[1];

    # Variables to store dot product

    # Calculating the dot product
    AB_BE = AB[0] * BE[0] + AB[1] * BE[1];
    AB_AE = AB[0] * AE[0] + AB[1] * AE[1];

    # Minimum distance from
    # point E to the line segment
    reqAns = 0;

    # Case 1
    if (AB_BE > 0):

        # Finding the magnitude
        y = E[1] - B[1];
        x = E[0] - B[0];
        reqAns = sqrt(x * x + y * y);

    # Case 2
    elif (AB_AE < 0):
        y = E[1] - A[1];
        x = E[0] - A[0];
        reqAns = sqrt(x * x + y * y);

    # Case 3
    else:

        # Finding the perpendicular distance
        x1 = AB[0];
        y1 = AB[1];
        x2 = AE[0];
        y2 = AE[1];
        mod = sqrt(x1 * x1 + y1 * y1);
        reqAns = abs(x1 * y2 - y1 * x2) / mod;

    return reqAns;
#-----------------------------------------



# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    frame=cv2.resize(frame,(1200,750))

    cv2.polylines(frame, [pts], True, (255, 0, 0), 2)

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, classes=[0]) # Tracking Car only
        #print(results)
        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        #print(boxes)
        track_ids = results[0].boxes.id.int().cpu().tolist()
        # Visualize the results on the frame
        #annotated_frame = results[0].plot()
        annotated_frame =frame
        countperson = 0
        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            # Draw Center point of object
            cv2.circle(annotated_frame, (int(x), int(y)), 1, (0, 255, 0), 2)
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)
            # Check Center point (x,y) in the polygon or not. ( In =1; Out =-1, On polygon =0)
            dist = cv2.pointPolygonTest(pts, (int(x), int(y)), False)
            if dist==1:
                cv2.circle(annotated_frame, (int(x), int(y)), 8, (0, 0, 255), 2)
                #if track_id not in crossed_objects:
                cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                                  (0, 255, 0), 2)
                countperson += 1
            # Draw the tracking lines
            #points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            #cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=3)


            # Check if the object crosses the line1

            A0 = [START[0].x, START[0].y];
            B0 = [END[0].x, END[0].y];
            E = [x, y];
            distance_0 = minDistance(A0, B0, E)
            if distance_0 < obcross:  # Assuming objects cross horizontally
                if track_id not in crossed_objects:
                    crossed_objects[track_id] = True
                # Annotate the object as it crosses the line
                cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),(0, 255, 0), 2)
                cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                              (0, 255, 0), 2)

        # Draw the line on the frame
        cv2.line(annotated_frame, (START[0].x, START[0].y), (END[0].x, END[0].y), (0, 255, 0), 2)
                # Write the count of objects on each frame
        count_text = f"Count_People: {len(crossed_objects)}"
        cv2.putText(annotated_frame, count_text, (121, 568), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(annotated_frame, "Xuan Ky Automation", (550, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 255, 0), 2)



        print(countperson)

        cv2.putText(annotated_frame,"countperson Region: "+ str(countperson),(250,400),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        # DELAY THE FRAME FOR CHECKING COUNTER
        time.sleep(0.1)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
           break

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()