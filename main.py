import cv2
import mediapipe as mp
import math


def calc_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

#Capture video feed
cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     cv2.imshow('webcam feed', frame)
    
# cap.release()
# cv2.destroyAllWindows()

#track hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

            hand_size = calc_distance(wrist, middle_tip)


            thumb_to_index = calc_distance(thumb_tip, index_tip) 
            thumb_to_wrist = calc_distance(thumb_tip, index_tip)
            index_to_wrist = calc_distance(thumb_tip, wrist)
            middle_to_wrist = calc_distance(middle_tip, wrist)
            ring_to_wrist = calc_distance(ring_tip, wrist)
            pinky_to_wrist = calc_distance(pinky_tip, wrist)

            fingers_to_wrist = [thumb_to_wrist, index_to_wrist, middle_to_wrist, ring_to_wrist, pinky_to_wrist]
            first_condition = all(distance <0.3 for distance in fingers_to_wrist)
            print(fingers_to_wrist)
            if first_condition:
                gesture = "closed fist"
            elif thumb_to_index > 0.4:
                gesture = "Open Hand"
            else:
                gesture = "unknown"
            cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Hand Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()