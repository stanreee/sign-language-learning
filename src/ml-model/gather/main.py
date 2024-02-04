from static import StaticClassifier
from dynamic import DynamicClassifier
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

static = StaticClassifier(hands)
dynamic = DynamicClassifier(hands)
classifiers = [static, dynamic]

while True:
    t = input("Enter type of handsign: static (0) or dynamic (1)")
    classifier = classifiers[int(t)]
    classifier.start()
