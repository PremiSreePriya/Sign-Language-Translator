# collect_data.py
import cv2
import mediapipe as mp
import pandas as pd
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

labels = ["A", "B", "C"]  # Add more labels as needed
SAVE_DIR = "data"

os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

data = []
label = input("Enter label for current gesture: ")

print("Press 's' to save frame, 'q' to quit")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            data.append(coords + [label])

    cv2.putText(frame, f"Label: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        print(f"Saved one sample for label '{label}'")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data)
df.to_csv("dataset.csv", mode='a', header=not os.path.exists("dataset.csv"), index=False)
