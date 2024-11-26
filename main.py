import cv2
import numpy as np
import mediapipe as mp
import math

def overlay_image(background, overlay):
    b, g, r, alpha = cv2.split(overlay)

    h, w = overlay.shape[:2]

    alpha_float = alpha.astype(float) / 255.0

    alpha_mask = np.stack((alpha_float,)*3, axis=-1)

    background[:h, :w] = (
        (1 - alpha_mask) * background[:h, :w] +
        alpha_mask * overlay[:, :, :3]
    )

    return background



handsDetector = mp.solutions.hands.Hands()

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

canvas = np.zeros((frame.shape[0], frame.shape[1], 4))

cv2.circle(canvas, (50, 50), 10, (255, 255, 255, 255), -1)

while (cap.isOpened()):
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break
    flipped = np.fliplr(frame)

    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

    results = handsDetector.process(flippedRGB)

    if results.multi_hand_landmarks is not None:
        x_tip = int(results.multi_hand_landmarks[0].landmark[8].x *
                    flippedRGB.shape[1])
        y_tip = int(results.multi_hand_landmarks[0].landmark[8].y *
                    flippedRGB.shape[0])
        x_avg = int(results.multi_hand_landmarks[0].landmark[12].x *
                    flippedRGB.shape[1])
        y_avg = int(results.multi_hand_landmarks[0].landmark[12].y *
                    flippedRGB.shape[0])
        x_big = int(results.multi_hand_landmarks[0].landmark[4].x *
                    flippedRGB.shape[1])
        y_big = int(results.multi_hand_landmarks[0].landmark[4].y *
                    flippedRGB.shape[0])

        if math.hypot(x_tip - x_big, y_tip - y_big) < 30:
            cv2.circle(canvas, ((x_tip + x_big) // 2, (y_tip + y_big) // 2), 10, (255, 255, 255, 255), -1)
        if math.hypot(x_big - x_avg, y_big - y_avg) < 30:
            cv2.circle(canvas, ((x_big + x_avg) // 2, (y_big + y_avg) // 2), 10, (0, 0, 0, 0), -1)

        cv2.circle(flippedRGB, (x_tip, y_tip), 10, (255, 0, 0), -1)
        cv2.circle(flippedRGB, (x_avg, y_avg), 10, (0, 255, 0), -1)
        cv2.circle(flippedRGB, (x_big, y_big), 10, (0, 0, 255), -1)
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)

    res = overlay_image(res_image, canvas)

    cv2.imshow("Hands", res)
