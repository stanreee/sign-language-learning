from static import StaticClassifier
from dynamic import DynamicClassifier
import cv2
import mediapipe as mp
from time import sleep

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

static = StaticClassifier(hands)
dynamic = DynamicClassifier(hands)
classifiers = [static, dynamic]

sleep(0.1)

while True:
    t = input("Enter type of handsign: static (0) or dynamic (1)\n")
    classifier = classifiers[int(t)]
    num_hands = input("Enter number of hands: 1 or 2\n")
    classifier.num_hands = int(num_hands)
    capture_type = input("Enter capture method, window (0) or command line (1) (window is easier, but if capture hotkey doesn't work use command line)\n")
    classifier.capture_type = int(capture_type)
    classifier.start()
