import cv2
import numpy as np
import mediapipe as mp

handsDetector = mp.solutions.hands.Hands()

cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break
    flipped = np.fliplr(frame) 

    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

    results = handsDetector.process(flippedRGB)

    if results.multi_hand_landmarks is not None:
        x_tip = int(results.multi_hand_landmarks[0].landmark[8].x * flippedRGB.shape[1])
        y_tip = int(results.multi_hand_landmarks[0].landmark[8].y * flippedRGB.shape[0])
        x_avg = int(results.multi_hand_landmarks[0].landmark[12].x * flippedRGB.shape[1])
        y_avg = int(results.multi_hand_landmarks[0].landmark[12].y * flippedRGB.shape[0])
        x_big = int(results.multi_hand_landmarks[0].landmark[4].x * flippedRGB.shape[1])
        y_big = int(results.multi_hand_landmarks[0].landmark[4].y * flippedRGB.shape[0])
        print(results.multi_hand_landmarks[0])
        print(results.multi_hand_landmarks[0])
        cv2.circle(flippedRGB,(x_tip, y_tip), 10, (255, 0, 0), -1)
        cv2.circle(flippedRGB,(x_avg, y_avg), 10, (0, 255, 0), -1)
        cv2.circle(flippedRGB,(x_big, y_big), 10, (0, 0, 255), -1)
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", res_image)
