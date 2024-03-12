import cv2
from classifier import Classifier
import os
import csv
from gather_util import extract_features, process_features

class StaticClassifier(Classifier):

    def __init__(self, hands) -> None:
        super().__init__("static", hands)
        pass

    def save(self, data, id, num_hands):
        for i in range(len(data)):
            data[i] = [id] + data[i]
        cur_dir = os.curdir
        fileName = str(self.name)
        with open(cur_dir + "/datasets/" + fileName + ".csv", 'a', encoding="UTF8", newline='') as f:
            writer = csv.writer(f, delimiter=',')
            for row in data:
                writer.writerow([i for i in row])
        print("Frames saved for id", id, ".")

    def capture(self, frame, frameNum, data):
        features, reflect, failed = extract_features(frame, self.hands, self.num_hands)
        if len(features) >= 21 if self.num_hands == 1 else 42 and not failed:
            features = process_features(features, reflect, shouldNormalize=True)
            data.append(features)
        # else:
        #     data, frameNum = self.forceEndCapture()
        if len(data) >= self.FRAME_CAP:
            data, frameNum = self.endCapture(data, frameNum)
        return (data, frameNum)
        
    
        