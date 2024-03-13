import cv2
import time
import os
import csv

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
            self.save(data, id, self.num_hands)
        data = []
        frameNum = 0
        return (data, frameNum)
    
    def forceEndCapture(self):
        self.capturing = False
        print("Capturing failed, frame missed. Try again.")
        data = []
        frameNum = 0
        return (data, frameNum)

    def start(self):
        # print("Press C to capture. If recording two hands, then a 3 second countdown will start when C is pressed before recording begins.")
        # while(input("Press c to capture. If recording two hands, then a 3 second countdown will start when c is pressed before recording begins.\n") != 'c'):
        #     pass
        while True:
            key = input("Press c to capture. If recording two hands, then a 3 second countdown will start when c is pressed before recording begins.\n")

            if key == 'c':
                print("Preparing to capture...")
                self.cap = cv2.VideoCapture(1)
                cv2.startWindowThread()
                cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
                data = []
                frameNum = 0
                startTime = 0
                countdown = False
                self.capturing = True
                print("CAPTURING 30 FRAMES")
                while self.capturing:
                    ret, frame = self.cap.read()

                    # if self.capturing:
                    #     frameNum += 1
                    #     data, frameNum = self.capture(frame, frameNum, data)

                    frameNum += 1
                    data, frameNum = self.capture(frame, frameNum, data)

                    cv2.imshow(self.name, frame)

                    if countdown:
                        deltaTime = time.time() - startTime
                        if deltaTime > 3:
                            countdown = False
                            self.capturing = True
                            startTime = 0
                            print("CAPTURING 30 FRAMES")

                    keys = cv2.waitKey(1) & 0xFF
                    # if keys == ord('c'):
                    #     if self.num_hands > 1:
                    #         if not self.capturing and not countdown:
                    #             startTime = time.time()
                    #             countdown = True
                    #             print("CAPTURING IN 3 SECONDS")
                    #     else:
                    #         print("CAPTURING 30 FRAMES")
                    #         self.capturing = True
                    # elif keys == ord('q'):
                    #     break
                            
                    if keys == ord('q'):
                        break
                self.cap.release()
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                cv2.waitKey(1)
                cv2.waitKey(1)
                cv2.waitKey(1)