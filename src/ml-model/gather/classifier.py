import cv2
import time
import os
import csv
from util import extract_features, process_features

class Classifier:

    FRAME_CAP = 30

    def __init__(self, name, hands) -> None:
        self.capturing = False
        self.name = name
        self.hands = hands
        pass

    def endCapture(self, data, frameNum):
        self.capturing = False
        print("END CAPTURING")
        save = input("Do you want to save these frames? (Y/N)")
        if save == "Y":
            id = input("Enter id for data.")
            self.save(data, id)
        data = []
        frameNum = 0
        return (data, frameNum)

    def capture(self, frame, frameNum, data):
        features = extract_features(frame, self.hands)
        if len(features) >= 21:
            features = process_features(features)
        if len(features) >= 21: 
            data.append(features)
        if frameNum >= self.FRAME_CAP:
            data, frameNum = self.endCapture(data, frameNum)
        # print(data, frameNum)
        return (data, frameNum)
            

    def start(self):
        self.cap = cv2.VideoCapture(1)
        cv2.startWindowThread()
        data = []
        frameNum = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if self.capturing:
                frameNum += 1
                data, frameNum = self.capture(frame, frameNum, data)

            cv2.imshow(self.name, frame)

            if cv2.waitKey(1) & 0xFF == ord('c'):
                print("CAPTURING 30 FRAMES")
                self.capturing = True
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)