import mediapipe as mp
import cv2
import math as m
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Font type
font = cv2.FONT_HERSHEY_SIMPLEX
# Colors
red = (50, 50, 255)
green = (127, 255, 0)
yellow = (0, 255, 255)
pink = (255, 0, 255)

try:
    # Read Train Data
    df = pd.read_csv('notebooks\\train data.csv')
    df.drop(columns='Unnamed: 0', inplace=True)

except:  # train.csv doesn't exist, so we make it exactly with code in notebook
    train_dict = {"Label": np.array(30*['Front']+30*['Side']),
                  "Shoulder_to_Hip_Ratio": np.array([1.82, 2.01, 1.84, 1.68, 1.75, 1.64, 1.77, 1.74, 1.59, 1.78, 1.71,
                                                     1.43, 1.61, 1.76, 1.63, 1.89, 1.76, 1.78, 1.77, 1.77, 1.94, 1.81,
                                                     1.72, 1.71, 1.66, 1.68, 1.79, 1.45, 1.82, 1.88, 1.11, 0.46, 1.41,
                                                     1.48, 0.61, 1.07, 1.40, 1.16, 1.37, 1.31, 1.45, 0.14, 0.99, 1.2,
                                                     0.8, 1.4, 1.25, 1.21, 1.13, 0.23, 0.70, 0.75, 0.83, 1.31, 0.44,
                                                     1.07, 0.94, 1.45, 1.13, 1.18]),
                  "Eye_to_Ear_Ratio": np.array([0.47, 0.58, 0.5, 0.5, 0.52, 0.49, 0.47, 0.55, 0.5, 0.47, 0.54, 0.51,
                                                0.56, 0.49, 0.6, 0.46, 0.61, 0.54, 0.53, 0.41, 0.53, 0.52, 0.5, 0.52,
                                                0.52, 0.42, 0.47, 0.48, 0.56, 0.55, 0.56, 0, 0.33, 0.41, 0.70, 0.54,
                                                0.56, 0.58, 0.54, 0.63, 0.49, 0.57, 0.54, 0.5, 0.51, 0.89, 0.58, 0.62,
                                                0, 0.6, 0.61, 0.6, 0.27, 0.58, 0.61, 0.57, 0.58, 0.74, 0.6, 0.54])}
    df = pd.DataFrame(train_dict)
    df.drop(index=df[(df.Label == 'Side') & (df.Shoulder_to_Hip_Ratio > 1.3)].index, inplace=True)
    df.drop(df[df.Eye_to_Ear_Ratio == 0].index, inplace=True)
    df.drop(index=34, inplace=True)
    df.drop(index=52, inplace=True)
    df.reset_index(inplace=True)
    df.drop(labels=['index'], axis=1, inplace=True)

# Divide X and y train data
X = df[['Shoulder_to_Hip_Ratio', 'Eye_to_Ear_Ratio']]
y = df['Label']

# Make Classifier and Train it
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X, y)


# Find Distance
def find_distance(x1, y1, x2, y2):
    return m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Find Angle
def find_angle(x1, y1, x2, y2):
    theta = m.acos((y1 ** 2 - y1 * y2) / (y1 * find_distance(x1, y1, x2, y2)))
    degree = int(180 / m.pi) * theta
    return degree


class Pose:

    def __init__(self):
        # Set Default mode
        self.mode = None

    def __call__(self, img):
        # Preprocessing
        pose = mp.solutions.pose
        model = pose.Pose()

        self.lmPose = pose.PoseLandmark
        self.h, self.w = img.shape[:2]

        # Change image to RGB, process and change it back to BGR
        self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.keypoints = model.process(self.image)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

        # Save keypoints' coordinates
        self.lm = self.keypoints.pose_landmarks

        # Make an attribute to save the output
        self.output = self.image.copy()
        self.process()
        return self.output

    def _process_left(self, l_ear_x, l_ear_y, r_shldr_x, r_shldr_y, l_shldr_x, l_shldr_y, l_hip_x, l_hip_y):
        # Calculate angles
        neck_inclination = find_angle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        torso_inclination = find_angle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

        # Draw landmarks
        cv2.circle(self.output, (l_shldr_x, l_shldr_y), 7, yellow, -1)
        cv2.circle(self.output, (l_ear_x, l_ear_y), 7, yellow, -1)

        cv2.circle(self.output, (r_shldr_x, r_shldr_y), 7, pink, -1)
        cv2.circle(self.output, (l_hip_x, l_hip_y), 7, yellow, -1)

        angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

        # Determine whether good posture or bad posture
        if neck_inclination < 40 and torso_inclination < 10:
            color = green
        else:
            color = red
            cv2.line(self.output, (l_shldr_x, l_shldr_y), (l_shldr_x, l_ear_y), green, 4)
            cv2.line(self.output, (l_hip_x, l_hip_y), (l_hip_x, l_shldr_y), green, 4)

        cv2.putText(self.output, angle_text_string, (10, 30), font, 0.9, color, 2)
        cv2.putText(self.output, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, color, 2)
        cv2.putText(self.output, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, color, 2)

        # Join landmarks
        cv2.line(self.output, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), color, 4)
        cv2.line(self.output, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), color, 4)

    def _process_right(self, r_ear_x, r_ear_y, r_shldr_x, r_shldr_y, l_shldr_x, l_shldr_y, r_hip_x, r_hip_y):
        # Calculate angles
        neck_inclination = find_angle(r_shldr_x, r_shldr_y, r_ear_x, r_ear_y)
        torso_inclination = find_angle(r_hip_x, r_hip_y, r_shldr_x, r_shldr_y)

        # Draw landmarks
        cv2.circle(self.output, (r_shldr_x, r_shldr_y), 7, yellow, -1)
        cv2.circle(self.output, (r_ear_x, r_ear_y), 7, yellow, -1)

        cv2.circle(self.output, (l_shldr_x, l_shldr_y), 7, pink, -1)
        cv2.circle(self.output, (r_hip_x, r_hip_y), 7, yellow, -1)

        angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

        # Determine whether good posture or bad posture
        if neck_inclination < 40 and torso_inclination < 10:
            color = green
        else:
            color = red
            cv2.line(self.output, (r_shldr_x, r_shldr_y), (r_shldr_x, r_ear_y), green, 4)
            cv2.line(self.output, (r_hip_x, r_hip_y), (r_hip_x, r_shldr_y), green, 4)

        cv2.putText(self.output, angle_text_string, (10, 30), font, 0.9, color, 2)
        cv2.putText(self.output, str(int(neck_inclination)), (r_shldr_x + 10, r_shldr_y), font, 0.9, color, 2)
        cv2.putText(self.output, str(int(torso_inclination)), (r_hip_x + 10, r_hip_y), font, 0.9, color, 2)

        # Join landmarks
        cv2.line(self.output, (r_shldr_x, r_shldr_y), (r_ear_x, r_ear_y), color, 4)
        cv2.line(self.output, (r_hip_x, r_hip_y), (r_shldr_x, r_shldr_y), color, 4)

    def _process_front(self, nose_x, nose_y, r_shldr_x, r_shldr_y, l_shldr_x, l_shldr_y):
        # Find the middle between right shoulder and left shoulder
        middle_x = int((l_shldr_x - r_shldr_x) / 2 + r_shldr_x)
        middle_y = int(abs(l_shldr_y - r_shldr_y) / 2 + min(r_shldr_y, l_shldr_y))

        # Calculate angles
        neck_inclination = int(find_angle(middle_x, middle_y, nose_x, nose_y))
        shoulder_inclination = int(find_angle(middle_x, middle_y, r_shldr_x, r_shldr_y))

        # Draw landmarks
        cv2.circle(self.output, (nose_x, nose_y), 7, yellow, -1)

        cv2.circle(self.output, (l_shldr_x, l_shldr_y), 7, yellow, -1)
        cv2.circle(self.output, (r_shldr_x, r_shldr_y), 7, yellow, -1)

        cv2.circle(self.output, (middle_x, middle_y), 7, pink, -1)

        # Determine whether good posture or bad posture
        if neck_inclination < 10 and abs(90 - shoulder_inclination) < 5:
            color = green
        else:
            color = red
            cv2.line(self.output, (middle_x, middle_y), (middle_x, nose_y), green, 4)
            cv2.line(self.output, (l_shldr_x, middle_y), (r_shldr_x, middle_y), green, 4)

        angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Shoulder : ' + str(int(shoulder_inclination))
        cv2.putText(self.output, angle_text_string, (10, 30), font, 0.9, color, 2)
        cv2.putText(self.output, str(neck_inclination), (nose_x, nose_y - 20), font, 0.9, color, 2)
        cv2.putText(self.output, str(shoulder_inclination), (middle_x, middle_y + 30), font, 0.9, color, 2)

        # Join landmarks
        cv2.line(self.output, (middle_x, middle_y), (nose_x, nose_y), color, 4)
        cv2.line(self.output, (r_shldr_x, r_shldr_y), (l_shldr_x, l_shldr_y), color, 4)

    def Front_or_Side(self, shldr_to_hip, eye_to_ear):
        # Make a DataFrame of our data
        data = pd.DataFrame({'Shoulder_to_Hip_Ratio': [shldr_to_hip], 'Eye_to_Ear_Ratio': [eye_to_ear]})

        # Predict and save mode
        result = classifier.predict(data)
        self.mode = result[0]

    def process(self):
        # Acquire the landmark coordinates
        # Right shoulder
        r_shldr_x = int(self.lm.landmark[self.lmPose.RIGHT_SHOULDER].x * self.w)
        r_shldr_y = int(self.lm.landmark[self.lmPose.RIGHT_SHOULDER].y * self.h)

        # Left shoulder
        l_shldr_x = int(self.lm.landmark[self.lmPose.LEFT_SHOULDER].x * self.w)
        l_shldr_y = int(self.lm.landmark[self.lmPose.LEFT_SHOULDER].y * self.h)

        # Nose
        nose_x = int(self.lm.landmark[self.lmPose.NOSE].x * self.w)
        nose_y = int(self.lm.landmark[self.lmPose.NOSE].y * self.h)

        # Right ear
        r_ear_x = int(self.lm.landmark[self.lmPose.RIGHT_EAR].x * self.w)
        r_ear_y = int(self.lm.landmark[self.lmPose.RIGHT_EAR].y * self.h)

        # Left ear
        l_ear_x = int(self.lm.landmark[self.lmPose.LEFT_EAR].x * self.w)
        l_ear_y = int(self.lm.landmark[self.lmPose.LEFT_EAR].y * self.h)

        # Right hip
        r_hip_x = int(self.lm.landmark[self.lmPose.RIGHT_HIP].x * self.w)
        r_hip_y = int(self.lm.landmark[self.lmPose.RIGHT_HIP].y * self.h)

        # Left hip
        l_hip_x = int(self.lm.landmark[self.lmPose.LEFT_HIP].x * self.w)
        l_hip_y = int(self.lm.landmark[self.lmPose.LEFT_HIP].y * self.h)

        # Right eye
        r_eye_x = int(self.lm.landmark[self.lmPose.RIGHT_EYE].x * self.w)
        r_eye_y = int(self.lm.landmark[self.lmPose.RIGHT_EYE].x * self.h)

        # Left eye
        l_eye_x = int(self.lm.landmark[self.lmPose.LEFT_EYE].x * self.w)
        l_eye_y = int(self.lm.landmark[self.lmPose.LEFT_EYE].x * self.h)

        # Calculate the needed distances
        eye_dif = int(find_distance(r_eye_x, r_eye_y, l_eye_x, l_eye_y))
        ear_dif = int(find_distance(r_ear_x, r_ear_y, l_ear_x, l_ear_y))
        shldr_dif = int(find_distance(r_shldr_x, r_shldr_y, l_shldr_x, l_shldr_y))
        hip_dif = int(find_distance(r_hip_x, r_hip_y, l_hip_x, l_hip_y))

        # Calculate needed Ratios
        try:
            shldr_to_hip = shldr_dif / hip_dif
        except ZeroDivisionError:  # It is definitely 'Side'
            self.mode = 'Side'
        else:  # If there was no exceptions, then...
            try:
                eye_to_ear = eye_dif / ear_dif
            except ZeroDivisionError:  # It is definitely 'Side'
                self.mode = 'Side'
            else:  # Again there were no exceptions, so predict mode
                self.Front_or_Side(shldr_to_hip, eye_to_ear)

        # Determine whether side or front
        if self.mode == 'Side':
            # Determine whether right side or left side
            if nose_x < l_ear_x and nose_x < r_ear_x:  # Left Side
                self._process_left(l_ear_x, l_ear_y, r_shldr_x, r_shldr_y, l_shldr_x, l_shldr_y, l_hip_x, l_hip_y)
            else:  # Right Side
                self._process_right(r_ear_x, r_ear_y, r_shldr_x, r_shldr_y, l_shldr_x, l_shldr_y, r_hip_x, r_hip_y)

        elif self.mode == 'Front':  # Front
            self._process_front(nose_x, nose_y, r_shldr_x, r_shldr_y, l_shldr_x, l_shldr_y)
