import cv2
import mediapipe as mp
import math


# Helper function to calculate Euclidean distance
def calc_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


# Capture video feed
cap = cv2.VideoCapture(0)

# Track hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract key landmarks
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

            # Calculate bounding box for normalization
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]
            hand_width = max(x_coords) - min(x_coords)
            hand_height = max(y_coords) - min(y_coords)
            hand_size = max(hand_width, hand_height)  # Normalizing factor

            # Normalize distances
            thumb_to_index = calc_distance(thumb_tip, index_tip) / hand_size
            thumb_to_wrist = calc_distance(thumb_tip, wrist) / hand_size
            index_to_wrist = calc_distance(index_tip, wrist) / hand_size
            middle_to_wrist = calc_distance(middle_tip, wrist) / hand_size
            ring_to_wrist = calc_distance(ring_tip, wrist) / hand_size
            pinky_to_wrist = calc_distance(pinky_tip, wrist) / hand_size

            # Check gesture conditions
            fingers_to_wrist = [
                thumb_to_wrist,
                index_to_wrist,
                middle_to_wrist,
                ring_to_wrist,
                pinky_to_wrist,
            ]
            fist_condition = all(distance < 0.88 for distance in fingers_to_wrist)
            open_hand_condition = thumb_to_index > 0.5 and all(distance > 0.4 for distance in fingers_to_wrist[1:])
            print(fingers_to_wrist)
            if fist_condition:
                gesture = "Closed Fist"
            elif open_hand_condition:
                gesture = "Open Hand"
            else:
                gesture = "Unknown"

            # Display gesture on frame
            cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display video feed
    cv2.imshow('Hand Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
