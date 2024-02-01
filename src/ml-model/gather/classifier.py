import cv2
import time
import os
import csv

class Classifier:

    def __init__(self, name, hands) -> None:
        self.capturing = False
        self.name = name
        self.hands = hands
        pass

    def start(self):
        self.cap = cv2.VideoCapture(1)
        cv2.startWindowThread()
        startTime = time.time()
        data = []
        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if self.capturing:
                features, frame = self.capture(frame)
                elapsed = time.time() - startTime
                if len(features) >= 21: 
                    data.append(features)
                if elapsed >= 2:
                    print("END CAPTURING")
                    save = input("Do you want to save these frames? (Y/N)")
                    if save == "Y":
                        id = input("Enter id for data.")
                        for i in range(len(data)):
                            data[i] = [id] + data[i]
                        cur_dir = os.curdir
                        with open(cur_dir + "/datasets/" + str(self.name) + ".csv", 'a', encoding="UTF8", newline='') as f:
                            writer = csv.writer(f, delimiter=',')
                            # data.sort(key=lambda x: x[0])
                            for row in data:
                                writer.writerow([i for i in row])
                        print("Frames saved for id", id, ".")
                    data = []
                    self.capturing = False

            cv2.imshow(self.name, frame)

            if cv2.waitKey(1) & 0xFF == ord('c'):
                print("CAPTURING FRAMES")
                self.capturing = True
                startTime = time.time()
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)