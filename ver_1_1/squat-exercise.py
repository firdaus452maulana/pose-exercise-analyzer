import cv2
import datetime
import os

from utils import get_mediapipe_pose, create_csv_files, create_dict_object
from squat_process import ProcessFrame
from thresholds import get_thresholds_beginner
from dictionary_feedback import get_feedback_words

# thresholds = get_thresholds_beginner()
thresholds = get_thresholds_beginner()

name_folder = os.path.join(str(datetime.date.today()), "set4")
if not os.path.exists(str(datetime.date.today())):
    os.mkdir(str(datetime.date.today()))
if not os.path.exists(name_folder):
    os.mkdir(name_folder)

live_process_frame = ProcessFrame(thresholds=thresholds, flip_frame=True, name_folder=name_folder)
# Initialize face mesh solution
pose = get_mediapipe_pose()

cap = cv2.VideoCapture("WIN_20240330_23_27_27_Pro.mp4")

while True:
    success, img = cap.read()
    if not success:
        break
    frame, _ = live_process_frame.process(img, pose)  # Process frame

    cv2.imshow("Image", frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

feedback_words = get_feedback_words()
create_dict_object(live_process_frame, name_folder)


create_csv_files(live_process_frame, feedback_words, name_folder)
