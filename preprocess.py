import ast
import pandas as pd
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import cv2
import mediapipe as mp
import numpy as np
from IPython.display import display, Image
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
# import mediapipe as mp
data_root = "./dataset"  # make this an argparse
train_dataset = ImageFolder(root=data_root)  # should go into a folder
# Initialize MediaPipe pose model
pose = mp_pose.Pose(static_image_mode=True,
                    model_complexity=2, min_detection_confidence=0.5)


def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        return landmarks
    else:
        return None


def extract_features(image_path):
    landmarks = extract_landmarks(image_path)
    if landmarks:
        # Extract specific landmarks (indexes from mediapipe documentation)
        selected_landmarks = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        ]
        # Extract x, y, z coordinates
        features = []

        for landmark in selected_landmarks:
            features.extend([landmark.x, landmark.y, landmark.z])
        return features, landmarks
    else:
        # If no landmarks are detected, return a vector of zeros
        return [0] * 30, []


POSE_CONNECTIONS = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER.value,
     mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
    (mp_pose.PoseLandmark.LEFT_SHOULDER.value,
     mp_pose.PoseLandmark.LEFT_ELBOW.value),
    (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
     mp_pose.PoseLandmark.RIGHT_ELBOW.value),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
     mp_pose.PoseLandmark.RIGHT_HIP.value),
    (mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value),
    (mp_pose.PoseLandmark.RIGHT_ELBOW.value,
     mp_pose.PoseLandmark.RIGHT_WRIST.value),
    (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value),
    (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value),
    (mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value),
]


def draw_selected_landmarks(image_path, landmarks):
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape
    unique_landmarks = set([i[0] for i in POSE_CONNECTIONS])
    s_landmarks = [landmarks[i] for i in unique_landmarks]
    print(s_landmarks)
    # for idx, landmark in enumerate(landmarks):
    for landmark in s_landmarks:

        if landmark.visibility >= 0.0:
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    for start_idx, end_idx in POSE_CONNECTIONS:
        if landmarks[start_idx].visibility > 0.5 and landmarks[end_idx].visibility >= 0.0:
            start_point = (int(landmarks[start_idx].x * image_width),
                           int(landmarks[start_idx].y * image_height))
            end_point = (int(landmarks[end_idx].x * image_width),
                         int(landmarks[end_idx].y * image_height))
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)

    return image


def prepare_data(image_label_list):
    features = []
    labels = []

    for image_path, label in image_label_list:
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            continue
        image_features, landmarks_ = extract_features(image_path)
        features.append(image_features)
        labels.append(label)
    return np.array(features), np.array(labels)


features, labels = prepare_data(train_dataset.imgs)
print("Features shape:", features.shape)
print("Labels shape:", labels.shape)

# convert to dataframe and save csv
csv_path = os.path.join(data_root, f"refined_ucf_action_landmarks.csv")
# write each to csv and shuffle so we get only one csv
df = pd.DataFrame({'features': features.tolist(), 'labels': labels})
df.to_csv(csv_path, index=False)
print(f'csv saved to {csv_path}')
# shuffle when creating train test split

# clean up all zero rows. i.e, all featureswith all zeros)

# Load the CSV file
file_path = csv_path
df = pd.read_csv(file_path, header=None)

# Function to check if a string represents a list of all zeros


def is_all_zeros(value):
    try:
        # Safely evaluate the string as a Python list
        lst = ast.literal_eval(value)
        if isinstance(lst, list) and all(x == 0.0 for x in lst):
            return True
    except (ValueError, SyntaxError):
        pass
    return False


# Filter out rows where the first column is a list of all zeros
filtered_df = df[~df[0].apply(is_all_zeros)]

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv("filtered_refined_ucf_landmarks.csv",
                   index=False, header=False)

print("Filtered file saved as 'filtered_file.csv'")
