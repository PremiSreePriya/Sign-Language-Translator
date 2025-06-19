import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
from collections import deque

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Capture webcam
cap = cv2.VideoCapture(0)

# Word prediction buffer
predicted_letter = ''
last_prediction_time = 0
word = ''
char_buffer = deque(maxlen=20)  # Stores recent characters

# Main loop
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

            if len(coords) == 63:
                predicted_letter = model.predict([coords])[0]
                current_time = time.time()

                # Only update every 1 second
                if current_time - last_prediction_time > 1:
                    last_prediction_time = current_time
                    char_buffer.append(predicted_letter)

                    # Combine into word if stable letters
                    if len(char_buffer) >= 3 and len(set(char_buffer)) == 1:
                        word += predicted_letter
                        char_buffer.clear()
    # Display word letter prediction
    cv2.putText(frame, f"Word: {predicted_letter}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


    # Reset word on pressing 'r'
    if cv2.waitKey(1) & 0xFF == ord('r'):
        word = ''

    cv2.imshow("Sign Language Translator", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
