import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # Use 0 for default camera or replace with video path

# Fingertips array with landmark IDs
fingertips = [4, 8, 12, 16, 20]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for tip_id in fingertips:
                x, y = int(hand_landmarks.landmark[tip_id].x * frame.shape[1]), int(hand_landmarks.landmark[tip_id].y * frame.shape[0])
                cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)

                # Check if the finger is folded
                finger_fold_status = []
                for i in range(1, len(fingertips)):
                    if hand_landmarks.landmark[fingertips[i]].x < hand_landmarks.landmark[fingertips[i - 1]].x:
                        cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                        finger_fold_status.append(True)
                    else:
                        finger_fold_status.append(False)

                # Check thumb gesture for like or dislike
                if all(finger_fold_status) and hand_landmarks.landmark[fingertips[0]].y < hand_landmarks.landmark[fingertips[0] - 2].y:
                    cv2.putText(frame, "LIKE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif all(finger_fold_status) and hand_landmarks.landmark[fingertips[0]].y > hand_landmarks.landmark[fingertips[0] - 2].y:
                    cv2.putText(frame, "DISLIKE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Sign Language Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
