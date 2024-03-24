import cv2
import mediapipe as mp
import os
from recognition_model import RecognitionModel
import time

cur_dir = os.getcwd()

dynamic = RecognitionModel([cur_dir + "/trained_models/dynamic_two_1.pt", cur_dir + "/trained_models/dynamic_two_2.pt"], "dynamic", debug=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

landmark_history = []
frameNum = 0

prev_input = None

capturing = False
countdown = False
startTime = 0

cap = cv2.VideoCapture(1)
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    features = []

    if countdown:
        deltaTime = time.time() - startTime
        if deltaTime > 3:
            countdown = False
            capturing = True
            startTime = 0
            print("CAPTURING FRAMES")

    if capturing and results.multi_hand_landmarks:
        for idx, hand in enumerate(results.multi_hand_landmarks):
            landmarks = hand
            for point in landmarks.landmark:
                x, y, z = int(point.x * frame.shape[1]), int(point.y * frame.shape[0]), int(point.z * frame.shape[1])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                features.append([point.x, point.y, point.z])
        if len(features) >= 21:
            landmark_history.append(features)
            if len(landmark_history) > 30:
                landmark_history.pop(0)
                result, confidence, input_vector = dynamic.evaluate(landmark_history, False, prev_input)
                prev_input = input_vector
                print(result, confidence)
                capturing = False
                landmark_history = []
    
    cv2.imshow("Hand Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        # capturing = True
        if not capturing and not countdown:
            startTime = time.time()
            countdown = True
            print("CAPTURING IN 3 SECONDS")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


    