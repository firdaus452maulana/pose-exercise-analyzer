import cv2
import csv
import datetime
import os

from utils import get_mediapipe_pose
from squat_process import ProcessFrame
from thresholds import get_thresholds_beginner
from dictionary_feedback import get_feedback_words

# thresholds = get_thresholds_beginner()
thresholds = get_thresholds_beginner()

name_folder = datetime.date.today()
if not os.path.exists(str(name_folder)):
    os.mkdir(str(name_folder))

live_process_frame = ProcessFrame(thresholds=thresholds, flip_frame=True, name_folder=name_folder)
# Initialize face mesh solution
pose = get_mediapipe_pose()

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break
    frame, _ = live_process_frame.process(img, pose)  # Process frame

    cv2.imshow("Image", frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

feedback_words = get_feedback_words()

state_rep = live_process_frame.state_tracker['STATE_REP']
data_feedback = live_process_frame.DATA_FEEDBACK
write_csv = [['repetition', 'state', 'feedback_1', 'feedback_2', 'feedback_3', 'feedback_4']]
for rep in range(len(state_rep)):
    # print(data_feedback)
    if state_rep[rep] == 'FAILED':
        write_csv.append([rep + 1, state_rep[rep], 'You are performing the squat imperfectly, you should lower the '
                                                   'squat until your thighs are in line with your knees.', '', '', ''])
    elif state_rep[rep] == 'IMPROPER':
        feedback = data_feedback[rep + 1]
        feedback_array = [-1, -1, -1, -1]
        for x in range(len(feedback)):
            feedback_array[x] = list(feedback)[x]
        write_csv.append([rep + 1, state_rep[rep], feedback_words[feedback_array[0]], feedback_words[feedback_array[1]],
                          feedback_words[feedback_array[2]], feedback_words[feedback_array[3]]])

    else:
        write_csv.append([rep + 1, state_rep[rep], 'You are strong, you are capable, and every squat is a testament '
                                                   'to your dedication and hard work. Keep moving, and let every '
                                                   'squat lift your spirit higher!', '', '', ''])

print(live_process_frame.DATA_FOR_CSV)
f = open('{}/data_feedback_squats.csv'.format(name_folder), 'w')

writer = csv.writer(f)

writer.writerows(write_csv)

f.close()
