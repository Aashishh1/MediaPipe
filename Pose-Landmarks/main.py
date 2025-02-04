import cv2
import mediapipe as mp

# Initialize the video capture object to access the webcam (default camera: 0)
cap = cv2.VideoCapture(0)

# Initialize MediaPipe's pose detection module
mpPose = mp.solutions.pose  # Access the pose solution
pose = mpPose.Pose()        # Create a Pose object for pose detection
mpDraw = mp.solutions.drawing_utils  # Utility for drawing landmarks and connections

while True:
    # Capture a single frame from the webcam
    _, img = cap.read()
    
    # Convert the frame from BGR (OpenCV default) to RGB (required by MediaPipe)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the RGB image to detect pose landmarks
    results = pose.process(imgRGB)

    # Check if pose landmarks are detected
    if results.pose_landmarks:
        poseLms = results.pose_landmarks  # Get pose landmarks data
        
        # Iterate through each detected landmark
        for id, lm in enumerate(poseLms.landmark):
            h, w, c = img.shape  # Get the dimensions of the frame
            cx, cy = int(lm.x * w), int(lm.y * h)  # Calculate pixel coordinates of the landmark
            
            # Highlight specific landmarks with colored circles
            if id == 21:  # Left thumb
                cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)  # Blue circle
            if id == 22:  # Right thumb
                cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)  # Green circle
            if id == 11:  # Left shoulder
                cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)  # Red circle
            if id == 12:  # Right shoulder
                cv2.circle(img, (cx, cy), 10, (0, 255, 255), cv2.FILLED)  # Yellow circle
        
        # Draw all detected pose landmarks and their connections on the frame
        mpDraw.draw_landmarks(img, poseLms, mpPose.POSE_CONNECTIONS)

    # Flip the image horizontally to act like a mirror
    img = cv2.flip(img, 1)
    
    # Display the frame in a window
    cv2.imshow('Image', img)
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
