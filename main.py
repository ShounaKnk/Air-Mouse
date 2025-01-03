import cv2
import mediapipe as mp
import math
import time
import pyautogui as pag

def calc_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

f = 0
last_pinch = 0
last_double_click =0
double_click = True
previous_psotions= []
bufferSize = 5
last_scroll = 0
drag_used = 0        
cooldown_time = .5
drag_active = False
drag_start_time = None
x, y, smooth_x, smooth_y = 0, 0, 0, 0
# alpha = 0.2
prev_x, prev_y = 0, 0
cursor_x, cursor_y = pag.position()
sensitivity = 9
screen_width, screen_height = pag.size()

def hand_stationary(index_tip, frame_width, frame_height):
    global prev_x, prev_y
    current_x = int(index_tip.x*frame_width)
    current_y = int(index_tip.y*frame_height)
    x_stable = abs(current_x - prev_x) < 5
    y_stable = abs(current_y - prev_y)< 5
    if x_stable and y_stable:
        return True
    else:
        return False
    


def cursor_move(index_tip, frame_width, frame_height, sensitivity):
    global cursor_x, cursor_y, prev_x, prev_y
    if not hand_stationary(index_tip, frame_width, frame_height):
        current_x = int(index_tip.x*frame_width)
        current_y = int(index_tip.y*frame_height)

        if prev_x is not None and prev_y is not None:
            del_x = (current_x - prev_x)*sensitivity
            del_y = (current_y - prev_y)*sensitivity
                                                                                    
                # cursor_x = cursor_x + alpha*del_x
                # cursor_y = cursor_y + alpha *del_y
            cursor_x += del_x
            cursor_y += del_y

            cursor_x = max(1, min(screen_width -2, cursor_x))
            cursor_y = max(1, min(screen_height -2, cursor_y,))
            pag.moveTo(int(cursor_x), int(cursor_y))
        prev_x, prev_y = current_x, current_y    

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
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
            ring_DIP = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]

            # Calculate bounding box for normalization
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]
            hand_width = max(x_coords) - min(x_coords)
            hand_height = max(y_coords) - min(y_coords)
            hand_size = max(hand_width, hand_height)

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
            # print(index_to_wrist)
            # mode = 1 if calc_distance(thumb_tip, ring_DIP) > 0.06 else 
            if calc_distance(thumb_tip, ring_DIP) >0.06:
                mode = 1
            else:
                if middle_to_wrist < 0.6:
                    mode = 3
                else: mode = 0
            gesture = ""
            current_time = time.time()
            if mode == 5:
                if thumb_to_index < 0.1 and middle_to_wrist > 0.6:
                    if current_time - last_pinch < 0.3:
                        if current_time - last_double_click < 0.5:
                            if not drag_active:
                                drag_active = True
                                gesture = "drag"
                                drag_used = 1
                                pag.mouseDown()
                                last_cursor_update = 0  # Initialize cursor update timing for drag
                            # Throttle cursor updates during drag
                            if current_time - last_cursor_update > 0.03:  # Update every 30ms
                                cursor_move(index_tip, frame.shape[1], frame.shape[0], sensitivity=3)
                                last_cursor_update = current_time
                        else:
                            gesture = "double click"
                            last_double_click = current_time
                        last_pinch = current_time
                    else:
                        gesture = "click"
                        pag.click()
                        last_pinch = current_time
                elif thumb_to_middle < 0.1 and index_to_wrist > 0.7:
                    if current_time - last_pinch > 1:  # Throttle right clicks
                        gesture = "right click"
                        pag.rightClick()
                        last_pinch = current_time
                else:
                    gesture = "unknown"
                    if drag_used:
                        pag.mouseUp()
                        drag_active = False
                        drag_used = 0

                # Print gesture if recognized
                if gesture != "unknown" and not drag_active:
                    print(gesture)

            elif mode == 0:
                if current_time - last_scroll >= cooldown_time:
                    if index_to_wrist >0.4 and middle_to_wrist > 0.4:
                        previous_psotions.append((index_tip.y, middle_tip.y))
                        if len(previous_psotions) > bufferSize:
                            previous_psotions.pop(0)
                        # print(previous_psotions)
                        avg_index_movement = sum(p[0] - previous_psotions[0][0] for p in previous_psotions)
                        avg_middle_movement = sum(p[1] - previous_psotions[0][1] for p in previous_psotions)
                        if avg_index_movement < -0.2 and avg_middle_movement < -0.2:
                            gesture = "scroll down"
                            last_scroll = current_time
                        elif avg_index_movement >0.2 and avg_middle_movement >0.2:
                            gesture = "scroll up"
                            last_scroll = current_time
                    else:
                        gesture = "unknown"
                else:
                    gesture = "Paused(repositioning)"
            elif mode ==3:
                print(middle_to_wrist)
                cursor_move(index_tip, frame.shape[1],frame.shape[0], sensitivity = 9)
            else:
                gesture = "invalid mode"
                previous_psotions.clear()
            # Display gesture on frame
            cv2.circle(frame, (x,y), 10, (0, 255, 0), -1)
            cv2.putText(frame, gesture, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, str(mode), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display video feed
    cv2.imshow('Hand Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
