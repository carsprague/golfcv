import numpy as np
import cv2 as cv
import mediapipe as mp
import nb_helpers
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation

# Create Videocapture object

# Using the live camera
# cap = cv.VideoCapture(0)

# Using a pre-recorded video
cap = cv.VideoCapture('Videos/swingc1.mp4')
cap_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
cap_length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

print("The capture length is ", cap_length, "frames long")

# Define codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('swingrory.avi', fourcc, 30.0, (cap_width, cap_height))

# Initialize MediaPipe stuff
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Initialize NumPy array for wireframe -> (3 dimensions) * (33 landmarks) * (frames)
wireframe = np.empty((3, 33, cap_length))
frame_num = 0

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
        #imgFLIP = cv.flip(imgRGB, 1)
        results = pose.process(imgRGB)
        #resultsw = pose.process(imgFLIP)
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        # cv.imshow("Image", img)
        out.write(img)
        landmarks = results.pose_world_landmarks.landmark
        for i in range(len(mpPose.PoseLandmark)):
            wireframe[:, i, frame_num] = (landmarks[i].x, landmarks[i].y, landmarks[i].z)
        frame_num += 1

        # q is exit key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# When everything is done, release the capture
cap.release()
out.release()
cv.destroyAllWindows()

# Create the wireframe video
fig = plt.figure()
fig.set_size_inches(5, 5, True)
ax = fig.add_subplot(projection = '3d')
anim = nb_helpers.time_animate(wireframe, fig, ax)
anim.save('wireframecarson.mp4', fps=30, dpi=300)
