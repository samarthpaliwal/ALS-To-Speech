import cv2
import mediapipe as mp
import pickle
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

DATA_PATH = 'data/'
letters = list("ABCDEFGHI KLMNOPQRSTUVWXY")  # skip J, Z
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
for l in letters:
    os.makedirs(os.path.join(DATA_PATH, l), exist_ok=True)

for letter in letters:
    print(f"Collecting for {letter}. Press 'c' to capture.")
    count = 0
    while count < 100:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        cv2.imshow('Capture', frame)
        if res.multi_hand_landmarks and cv2.waitKey(1) & 0xFF == ord('c'):
            lm = res.multi_hand_landmarks[0].landmark
            vec = []
            for p in lm:
                vec.extend([p.x, p.y, p.z])
            with open(f"{DATA_PATH}/{letter}/{count}.pkl", 'wb') as f:
                pickle.dump(vec, f)
            count += 1
    print(f"Done {letter}")
cap.release()
