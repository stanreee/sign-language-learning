import mediapipe as mp
import cv2
import os
import numpy as np
import csv

# process landmark features
# in particular:
#   - designate a landmark as the base landmark, calculate relative x and y coordinates on other landmarks from the base landmark
#     this accommodates for different hand positions by having coordinates be relative to base landmark position and not camera position
def process_features(features):
    base_x, base_y = 0, 0
    for feature in features:
        if base_x == 0 and base_y == 0:
            base_x = feature[0]
            base_y = feature[1]
        feature[0] = base_x - feature[0]
        feature[1] = base_y - feature[1]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cur_dir = os.getcwd()

image_path = cur_dir + "/test/datasets/train/images/"
image_gens = cur_dir + "/test/datasets/anchors/"
image_data = cur_dir + "/test/datasets/train/csv/"

os.makedirs(image_gens, exist_ok=True)

data = []

for idx, ip in enumerate(os.listdir(image_path)):
    if ip.endswith(".jpg"):
        image = cv2.imread(image_path + ip)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        letter = float(ord(ip[0].lower()) - 97)

        results = hands.process(image)

        features = []

        row_data = []

        row_data.append(letter)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            for point in landmarks.landmark:
                x, y, z = int(point.x * image.shape[1]), int(point.y * image.shape[0]), int(point.z * image.shape[1])
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                features.append([point.x, point.y])
        
        process_features(features)

        flattened = np.array(features).flatten()

        for feature in flattened:
            row_data.append(feature)

        row = row_data

        if len(row) > 2:
            data.append(row)

        percentage = ((idx + 1) / len(os.listdir(image_path))) * 100

        print("collected data for", ip, str(percentage) + "%")

with open(image_data + "data.csv", 'w', encoding="UTF8", newline='') as f:
    writer = csv.writer(f, delimiter=',')
    data.sort(key=lambda x: x[0])
    for row in data:
        writer.writerow([i for i in row])