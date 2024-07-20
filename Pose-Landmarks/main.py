import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

while True:
    _, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        poseLms = results.pose_landmarks
        for id, lm in enumerate(poseLms.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)

            if id == 21: # left thumb
                cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
            if id == 22: # right thumb
                cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
            if id == 11: # left shoulder
                cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
            if id == 12: # right shoulder
                cv2.circle(img, (cx, cy), 10, (0, 255, 255), cv2.FILLED)
    
        mpDraw.draw_landmarks(img, poseLms, mpPose.POSE_CONNECTIONS)

    img = cv2.flip(img, 1) # to see image like mirror
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()