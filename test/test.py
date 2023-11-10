import mediapipe as mp
import cv2
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cur_dir = os.getcwd()

image_path = cur_dir + "/test/datasets/train/images/"
image_gens = cur_dir + "/test/datasets/anchors/"

os.makedirs(image_gens, exist_ok=True)

for ip in os.listdir(image_path):
    if ip.endswith(".jpg"):
        image = cv2.imread(image_path + ip)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                for point in landmarks.landmark:
                    x, y, z = int(point.x * image.shape[1]), int(point.y * image.shape[0]), int(point.z * image.shape[1])
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        
        print("writing to", image_gens + ip)
        cv2.imwrite(image_gens + ip, image)