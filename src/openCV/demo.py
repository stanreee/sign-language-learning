# imports
import cv2 as cv
import numpy as np

# verify that your OpenCV works
img = cv.imread('../assets/Toronto_Blue_Jays_logo.png')

cv.imshow('Display window', img)
k = cv.waitKey(0)
cv.destroyAllWindows()

# basic live video capturing (requires a webcam)
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()