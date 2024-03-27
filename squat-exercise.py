import cv2
import csv

from utils import get_mediapipe_pose
from squat_process import ProcessFrame
from thresholds import get_thresholds_beginner

# thresholds = get_thresholds_beginner()
thresholds = get_thresholds_beginner()

live_process_frame = ProcessFrame(thresholds=thresholds, flip_frame=True)
# Initialize face mesh solution
pose = get_mediapipe_pose()

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    frame, _ = live_process_frame.process(img, pose)  # Process frame

    cv2.imshow("Image", frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

print(live_process_frame.DATA_FOR_CSV)
f = open('data_squats_5.csv', 'w')

writer = csv.writer(f)

writer.writerows(live_process_frame.DATA_FOR_CSV)

f.close()