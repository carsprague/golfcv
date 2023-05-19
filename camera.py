import numpy as np
import cv2 as cv
import mediapipe as mp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import time
import nb_helpers

# Using the live camera
# cap = cv.VideoCapture(0)

# Using a pre-recorded video
cap = cv.VideoCapture('Videos/1.mp4')
length = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) # find length of video
frame_num = 0

# Create 3x33xlength NumPy array (3d * 33 landmarks * frame counth)
data = np.empty((3, 33, length))

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Define codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('C:/Users/carsp/golfcv/video.mp4', fourcc, 30.0, (640, 480))

# Something went wrong opening camera
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Loop for while capture is open
while cap.isOpened():
    # Capture img by img
    ret, img = cap.read()
    
    # if img is read correctly ret is True
    if not ret:
        print("Can't recieve img (stream end?). Exiting... ")
        break

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB) # convert from BGR to RGB
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv.circle(img, (cx,cy), 3, (0, 0, 255), cv.FILLED)

        out.write(img)          

    # We also will create a wireframe from the NumPy array we created earlier
    landmarks = results.pose_world_landmarks.landmark
    for i in range(len(mpPose.PoseLandmark)):
        data[:, i, frame_num] = (landmarks[i].x, landmarks[i].y, landmarks[i].z)
    frame_num+=1

    # q is exit key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
out.release()
cv.destroyAllWindows()

# Create and save wireframe
fig = plt.figure()
fig.set_size_inches(5, 5, True)
ax = fig.add_subplot(projection='3d')

anim = nb_helpers.time_animate(data, fig, ax)
anim.save('wireframe.mp4', fps=30, dpi=300)