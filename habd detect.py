import cv2
import mediapipe as mp

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Open webcam
cap = cv2.VideoCapture(0)

# Finger tip landmark IDs
finger_tips = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    if not success:
        break

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = []

            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = img.shape
                lm_list.append((id, int(lm.x * w), int(lm.y * h)))

            # Count fingers
            fingers = []

            # Thumb
            if lm_list[finger_tips[0]][1] > lm_list[finger_tips[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Other 4 fingers
            for i in range(1, 5):
                if lm_list[finger_tips[i]][2] < lm_list[finger_tips[i] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            finger_count = fingers.count(1)

            # Display finger count
            cv2.putText(img, f'Fingers: {finger_count}', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw landmarks
            mp_draw.draw_landmarks(
                img, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Hand Gesture Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
