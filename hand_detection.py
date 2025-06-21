import cv2
import mediapipe as mp

# 1. Set up MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,                 # detect one hand
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# 2. Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No frame captured—check camera.")
        break

    frame = cv2.flip(frame, 1)  # mirror image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 3. Process frame and find hands
    result = hands.process(rgb)

    # 4. Draw landmarks if a hand is found
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 5. Show the output
    cv2.imshow("ASL Hand Detection", frame)

    # 6. Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
