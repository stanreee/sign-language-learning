from classifier import Classifier
import os
import csv
from sklearn.decomposition import PCA
import numpy as np
from gather_util import extract_features, process_features, landmark_history_preprocess, normalize_landmark_history

class DynamicClassifier(Classifier):
    def __init__(self, hands) -> None:
        super().__init__("dynamic", hands)
        self.base_coords = None
        self.hand_data = [[], []]
        pass

    def endCapture(self, frameNum):
        self.capturing = False
        print("END CAPTURING")
        save = input("Do you want to save these frames? (Y/N)\n")
        if save == "Y" or save == 'y':
            id = input("Enter id for data.\n")
            self.save(self.hand_data, id, self.num_hands)
        data = []
        frameNum = 0
        return (data, frameNum)

    def save(self, hand_data, id, num_hands):
        for i in range(self.num_hands):
            data = hand_data[i]
            compressed, base = normalize_landmark_history(data, False, base_coords=self.base_coords)
            self.base_coords = base

            dataToSave = [id] + compressed

            fileName = str(self.name)
            if num_hands > 1:
                fileName = str(self.name) + "_two_" + (i + 1)
            cur_dir = os.curdir
            with open(cur_dir + "/datasets/" + fileName + ".csv", 'a', encoding="UTF8", newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(dataToSave)
        print("Frames saved for id", str(id) + ".")

    # def save(self, data, id, num_hands):
    #     print(len(data), len(data[0]))
    #     compressed = normalize_landmark_history(data, False, num_hands)
    #     print(len(compressed))
    #     dataToSave = [id] + compressed
        
    #     # save to file
    #     fileName = str(self.name)
    #     if num_hands > 1: fileName += "_2"
    #     cur_dir = os.curdir
    #     with open(cur_dir + "/datasets/" + fileName + ".csv", 'a', encoding="UTF8", newline='') as f:
    #         writer = csv.writer(f, delimiter=',')
    #         writer.writerow(dataToSave)
    #     print("Frames saved for id", str(id) + ".")
    
    def capture(self, frame, frameNum, data):
        features, reflect, failed = extract_features(frame, self.hands, self.num_hands)
        if len(features) >= 21 if self.num_hands == 1 else 42 and not failed:
            for i in range(self.num_hands):
                hand_features = features[:21] if i == 0 else features[21:]
                self.hand_data[i].append(hand_features)
                # data.append(features)
        # else:
        #     data, frameNum = self.forceEndCapture()
        if len(data) >= self.FRAME_CAP:
            data, frameNum = self.endCapture(self.hand_data, frameNum)
            self.base_coords = None
            self.hand_data = [[], []]
        return (data, frameNum)