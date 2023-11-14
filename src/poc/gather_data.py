import mediapipe as mp
import cv2
import os
import numpy as np
import csv
from util import process_features

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cur_dir = os.getcwd()

letters_to_process = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

image_path = cur_dir + "/src/poc/datasets/train/images/"
image_gens = cur_dir + "/src/poc/datasets/anchors/"
image_data = cur_dir + "/src/poc/datasets/train/csv/"

os.makedirs(image_gens, exist_ok=True)

data = []

for idx, ip in enumerate(os.listdir(image_path)):
    if ip.endswith(".jpg") and ip[0].lower() in letters_to_process:
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

        if len(features) == 0:
            continue
        
        features = process_features(features)

        row = row_data + features

        if len(row) > 2:
            data.append(row)

        percentage = ((idx + 1) / len(os.listdir(image_path))) * 100

        print("collected data for", ip, str(percentage) + "%")

with open(image_data + "data.csv", 'w', encoding="UTF8", newline='') as f:
    writer = csv.writer(f, delimiter=',')
    data.sort(key=lambda x: x[0])
    for row in data:
        writer.writerow([i for i in row])