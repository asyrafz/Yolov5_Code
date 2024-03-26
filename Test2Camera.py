import numpy as np
import cv2

cams_test = 500
for i in range(0, cams_test):
    cap = cv2.VideoCapture(i)
    test, frame = cap.read()
    print("i : "+str(i)+" /// result: "+str(test))

video_capture_0 = cv2.VideoCapture(0)
video_capture_1 = cv2.VideoCapture(1)

while True:
    ret0, frame0 = video_capture_0.read()
    ret1, frame1 = video_capture_1.read()

    #Check if the frames are correctly read
    if not ret0:
        print("Error: Could not read the frames ret0.")
        break

    if not ret1:
        #print("Error: Could not read the frames ret1.")
        break

    cv2.imshow('Cam 0', frame)
    cv2.imshow('Cam 1', frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture_0.release()
video_capture_1.release()
cv2.destroyAllWindows()
