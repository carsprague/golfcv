import numpy as np
import cv2 as cv
import mediapipe as mp

# Using the live camera
# cap = cv.VideoCapture(0)

# Using a pre-recorded video
cap = cv.VideoCapture('Videos/1.mp4')
cap_width = int(cap.get(3))
cap_height = int(cap.get(4))

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Define codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('output.avi', fourcc, 20.0, (cap_width, cap_height))

# Something went wrong opening camera
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Loop for while capture is open
while cap.isOpened():
    # Capture img
    success, img = cap.read()

    if success:
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB) # convert from BGR to RGB
        results = pose.process(imgRGB)
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        cv.imshow("Image", img)
        out.write(img)

        # q is exit key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    else:
        break

# When everything is done, release the capture
cap.release()
out.release()
cv.destroyAllWindows()
