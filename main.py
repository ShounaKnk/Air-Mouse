import cv2
import mediapipe as mp
import math
import time


# Helper function to calculate Euclidean distance
def calc_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


# Capture video feed
cap = cv2.VideoCapture(0)

# Track hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
last_pinch = 0
f = 0
previous_psotions= []
bufferSize = 5

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
            ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]

            # Calculate bounding box for normalization
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]
            hand_width = max(x_coords) - min(x_coords)
            hand_height = max(y_coords) - min(y_coords)
            hand_size = max(hand_width, hand_height)  # Normalizing factor

            # Normalize distances
            thumb_to_wrist = calc_distance(thumb_tip, wrist) / hand_size
            index_to_wrist = calc_distance(index_tip, wrist) / hand_size
            middle_to_wrist = calc_distance(middle_tip, wrist) / hand_size
            ring_to_wrist = calc_distance(ring_tip, wrist) / hand_size
            pinky_to_wrist = calc_distance(pinky_tip, wrist) / hand_size

            thumb_to_index = calc_distance(thumb_tip, index_tip) / hand_size
            thumb_to_middle = calc_distance(thumb_tip, middle_tip) / hand_size

            # Check gesture conditions
            fingers_to_wrist = [
                thumb_to_wrist,
                index_to_wrist,
                middle_to_wrist,
                ring_to_wrist,
                pinky_to_wrist,
            ]

            closed_fingers = all(distance < 0.88 for distance in fingers_to_wrist)
            open_hand_condition = thumb_to_index > 0.5 and all(distance > 0.4 for distance in fingers_to_wrist[1:])
            print(calc_distance(thumb_tip, ring_pip))
            mode = 1 if calc_distance(thumb_tip, ring_pip) > 0.06 else 0
            gesture = ""
            current_time = time.time()
            if mode == 1:
                if thumb_to_index <0.2 and middle_to_wrist >0.8:
                    if current_time - last_pinch <0.3:
                        f = 1
                    else:
                        f=0
                        gesture = "click"
                    last_pinch = current_time
                    time.sleep(0.15)
                    if f == 1:
                        gesture = "double click"
                elif thumb_to_middle < 0.2 and thumb_to_index>0.3:
                    gesture = "right click"
                else:
                    gesture = "unknown"
            elif mode == 0:
                if closed_fingers and thumb_to_index > 0.3 and thumb_to_middle > 0.3:
                    if index_to_wrist >0.4 and middle_to_wrist > 0.4:
                        previous_psotions.append((index_tip.y, middle_tip.y))
                        if len(previous_psotions) > bufferSize:
                            previous_psotions.pop(0)
                        
                        avg_index_movement = sum(p[0] - previous_psotions[0][0] for p in previous_psotions)
                        avg_middle_movement = sum(p[1] - previous_psotions[0][1] for p in previous_psotions)
                        
                        if avg_index_movement < -0.2 and avg_middle_movement < -0.2:
                            gesture = "scroll down"
                        elif avg_index_movement >0.2 and avg_middle_movement >0.2:
                            gesture = "scroll up"
                        else:
                            gesture = "unknown"
            else:
                gesture = "invalid mode"
                previous_psotions.clear()
            # Display gesture on frame
            cv2.putText(frame, gesture, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display video feed
    cv2.imshow('Hand Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
