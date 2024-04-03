import base64
import cv2
import mediapipe as mp
import numpy as np
import csv


def _image_to_base64(img):
    _, im_arr = cv2.imencode('.jpg', img)
    frame_byte = im_arr.tobytes()
    frame_b64 = base64.b64encode(frame_byte)
    return frame_b64


def draw_rounded_rect(img, rect_start, rect_end, corner_width, box_color):
    x1, y1 = rect_start
    x2, y2 = rect_end
    w = corner_width

    # draw filled rectangles
    cv2.rectangle(img, (x1 + w, y1), (x2 - w, y1 + w), box_color, -1)
    cv2.rectangle(img, (x1 + w, y2 - w), (x2 - w, y2), box_color, -1)
    cv2.rectangle(img, (x1, y1 + w), (x1 + w, y2 - w), box_color, -1)
    cv2.rectangle(img, (x2 - w, y1 + w), (x2, y2 - w), box_color, -1)
    cv2.rectangle(img, (x1 + w, y1 + w), (x2 - w, y2 - w), box_color, -1)

    # draw filled ellipses
    cv2.ellipse(img, (x1 + w, y1 + w), (w, w),
                angle=0, startAngle=-90, endAngle=-180, color=box_color, thickness=-1)

    cv2.ellipse(img, (x2 - w, y1 + w), (w, w),
                angle=0, startAngle=0, endAngle=-90, color=box_color, thickness=-1)

    cv2.ellipse(img, (x1 + w, y2 - w), (w, w),
                angle=0, startAngle=90, endAngle=180, color=box_color, thickness=-1)

    cv2.ellipse(img, (x2 - w, y2 - w), (w, w),
                angle=0, startAngle=0, endAngle=90, color=box_color, thickness=-1)

    return img


def draw_dotted_line(frame, lm_coord, start, end, line_color):
    pix_step = 0

    for i in range(start, end + 1, 8):
        cv2.circle(frame, (lm_coord[0], i + pix_step), 2, line_color, -1, lineType=cv2.LINE_AA)

    return frame


def draw_text(
        img,
        msg,
        width=8,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        pos=(0, 0),
        font_scale=1,
        font_thickness=2,
        text_color=(0, 255, 0),
        text_color_bg=(0, 0, 0),
        box_offset=(20, 10),
):
    offset = box_offset
    x, y = pos
    text_size, _ = cv2.getTextSize(msg, font, font_scale, font_thickness)
    text_w, text_h = text_size
    rec_start = tuple(p - o for p, o in zip(pos, offset))
    rec_end = tuple(m + n - o for m, n, o in zip((x + text_w, y + text_h), offset, (25, 0)))

    # img = draw_rounded_rect(img, rec_start, rec_end, width, text_color_bg)

    cv2.putText(
        img,
        msg,
        (int(rec_start[0] + 6), int(y + text_h + font_scale - 1)),
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )

    return text_size


def find_angle(p1, p2, ref_pt=np.array([0, 0])):
    p1_ref = p1 - ref_pt
    p2_ref = p2 - ref_pt

    cos_theta = (np.dot(p1_ref, p2_ref)) / (1.0 * np.linalg.norm(p1_ref) * np.linalg.norm(p2_ref))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    degree = int(180 / np.pi) * theta

    return int(degree)


def get_landmark_array(pose_landmark, key, frame_width, frame_height):
    denorm_x = int(pose_landmark[key].x * frame_width)
    denorm_y = int(pose_landmark[key].y * frame_height)

    return np.array([denorm_x, denorm_y])


def get_landmark_features(kp_results, dict_features, feature, frame_width, frame_height):
    if feature == 'nose':
        return get_landmark_array(kp_results, dict_features[feature], frame_width, frame_height)

    elif feature == 'left' or 'right':
        shldr_coord = get_landmark_array(kp_results, dict_features[feature]['shoulder'], frame_width, frame_height)
        elbow_coord = get_landmark_array(kp_results, dict_features[feature]['elbow'], frame_width, frame_height)
        wrist_coord = get_landmark_array(kp_results, dict_features[feature]['wrist'], frame_width, frame_height)
        hip_coord = get_landmark_array(kp_results, dict_features[feature]['hip'], frame_width, frame_height)
        knee_coord = get_landmark_array(kp_results, dict_features[feature]['knee'], frame_width, frame_height)
        ankle_coord = get_landmark_array(kp_results, dict_features[feature]['ankle'], frame_width, frame_height)
        foot_coord = get_landmark_array(kp_results, dict_features[feature]['foot'], frame_width, frame_height)

        return shldr_coord, elbow_coord, wrist_coord, hip_coord, knee_coord, ankle_coord, foot_coord

    else:
        raise ValueError("feature needs to be either 'nose', 'left' or 'right")


def get_mediapipe_pose():
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return pose


def create_csv_files(data, feedback_dict, name_folder):
    state_rep = data.state_tracker['STATE_REP']
    data_feedback = data.DATA_FEEDBACK
    write_csv = [['repetition', 'state', 'feedback_1', 'feedback_1_img', 'feedback_2', 'feedback_2_img', 'feedback_3',
                  'feedback_3_img', 'feedback_4', 'feedback_4_img']]
    for rep in range(len(state_rep)):
        # print(data_feedback)
        if state_rep[rep] == 'FAILED':
            write_csv.append([rep + 1, state_rep[rep], 'You are performing the squat imperfectly, you should lower the '
                                                       'squat until your thighs are in line with your knees.', '', '',
                              ''])
        elif state_rep[rep] == 'IMPROPER':
            feedback = data_feedback[rep + 1]
            print(feedback)
            feedback_array = [-1, -1, -1, -1]
            for x in range(len(feedback)):
                feedback_array[x] = list(feedback)[x]
            write_csv.append(
                [rep + 1, state_rep[rep], feedback_dict[feedback_array[0]], feedback_dict[feedback_array[1]],
                 feedback_dict[feedback_array[2]], feedback_dict[feedback_array[3]]])

        else:
            write_csv.append(
                [rep + 1, state_rep[rep], 'You are strong, you are capable, and every squat is a testament '
                                          'to your dedication and hard work. Keep moving, and let every '
                                          'squat lift your spirit higher!', '', '', ''])

    f = open('{}/data_feedback_squats.csv'.format(name_folder), 'w')

    writer = csv.writer(f)

    writer.writerows(write_csv)

    f.close()


def create_dict_object(data, name_folder):
    state_rep = data.state_tracker['STATE_REP']
    data_feedback = data.DATA_FEEDBACK
    data_obj = []
    for rep in range(len(state_rep)):
        repetition_obj = {'repetition': rep, 'state': state_rep[rep]}
        feedback_arr = []
        for feedback in data_feedback[rep + 1]:
            image = cv2.imread(f'{name_folder}/{rep}_{feedback}.jpg')
            img_b64 = _image_to_base64(image)
            feedback_arr.append({'feedback': feedback, 'img_64': img_b64})
        repetition_obj['feedbacks'] = feedback_arr
        data_obj.append(repetition_obj)

    with open(f'{name_folder}/data_feedback_squats.txt', "w") as text_file:
        text_file.write(str(data_obj))
