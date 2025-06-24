import cv2 as cv
import mediapipe as mp
import time
import os
import pygame
import numpy as np 

pygame.mixer.init()
shutter_sound=pygame.mixer.Sound("shutter.wav")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw=mp.solutions.drawing_utils

cap = cv.VideoCapture(0)
photo_captured= False
countdown_start_time= None
countdown_duration= 3

def is_fist(hand_landmarks):
    tips=[8,12,16,20]
    joints=[6,10,14,18]
    wrist_y = hand_landmarks.landmark[0].y

    for tip, joint in zip(tips, joints):
        tip_y= hand_landmarks.landmark[tip].y
        joint_y= hand_landmarks.landmark[joint].y

        if tip_y < joint_y or wrist_y <  joint_y:
            return False

    return True

countdown_triggered= False

while True:
    ret, frame= cap.read()
    if not ret:
        break

    frame= cv.flip(frame,1)

    rgb_frame= cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result= hands.process(rgb_frame)

    fist_detected= False

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            if is_fist(hand_landmarks):
                fist_detected= True
                break

    if fist_detected and not countdown_triggered and not photo_captured:
             countdown_triggered= True
             countdown_start_time= time.time()
             print("Countdown has Started")

    if countdown_triggered:
        elapsed= time.time()-countdown_start_time
        remaining= int(countdown_duration-elapsed)+1

        if remaining > 0:
            cv.putText(frame,f"{remaining}",(300,200), cv.FONT_HERSHEY_COMPLEX, 5, (0,0,255), 10)
            cv.putText(frame,"Smile Big !", (30,50), cv.FONT_HERSHEY_COMPLEX, 1.2, (0,0,0), 3)
        else:
            if not photo_captured:
                shutter_sound.play()
                white= 255*np.ones_like(frame)
                cv.imshow("ShutterSnap", white)
                cv.waitKey(200)
                timestamp= time.strftime("%Y%m%d-%H%M%S")
                clean_frame = frame.copy()
                script_dir= os.path.dirname(os.path.abspath(__file__))
                filename= os.path.join(script_dir,f"photo_{timestamp}.jpg")                  
                cv.imwrite(filename, clean_frame)
                print(f"Photo saved as photo_{timestamp}")
                photo_captured= True
                countdown_start_time = None
    if not fist_detected and photo_captured:
        countdown_triggered=  False
        countdown_start_time = None
        photo_captured= False
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv.imshow("ShutterSnap", frame)
    key = cv.waitKey(1)

    if cv.getWindowProperty("ShutterSnap", cv.WND_PROP_VISIBLE)< 1:
        print("Window Closed")
        break

    if key & 0xFF==ord('q'):
        print("Window Closed")
        break

cap.release()
cv.destroyAllWindows()