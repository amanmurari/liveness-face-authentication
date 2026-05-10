from configs import *
import torch
import cv2
import numpy as np
from random import randint
from questions import *
from liveness import detect_liveness,is_exists





COUNTER, TOTAL = 0,0
counter_ok_questions = 0
counter_ok_consecutives = 0
limit_consecutives = 3
limit_questions = 6
counter_try = 0
limit_try = 50 



def show_image(cam,text,color = (0,0,255)):
    ret, im = cam.read()
    im = cv2.flip(im, 1)
    cv2.putText(im,text,(10,50),cv2.FONT_HERSHEY_COMPLEX,1,color,2)
    return im

cam = cv2.VideoCapture(0)
cv2.namedWindow('liveness_detection', cv2.WINDOW_NORMAL)


for i in range(limit_questions):
    question=question_bank(randint(0,5))
    COUNTER, TOTAL = 0, 0
    counter_ok_consecutives = 0
    print("Question : ",question)
    im = show_image(cam,question)
    cv2.imshow('liveness_detection',im)
    prev_total = TOTAL
    if cv2.waitKey(1) &0xFF == ord('q'):
        break

    for j in range(limit_try):
        ret, frame = cam.read()
        if not ret:
            print("Error: Failed to capture frame.")
            continue
        frame = cv2.flip(frame, 1)

        im = frame.copy()
        cv2.putText(im, question, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('liveness_detection', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if question == "blink eyes":
            
            COUNTER, TOTAL = detect_liveness(frame, TOTAL, COUNTER, question)
            is_blink = TOTAL - prev_total
            if is_blink:
                challenge_res = "pass"
            else:
                challenge_res = "fail"
        else:
            out = detect_liveness(frame, TOTAL, COUNTER, question)
            if out:
                challenge_res = "pass"
            else:
                challenge_res = "fail"


        if challenge_res == "pass":
            im = show_image(cam,question+" : ok")
            cv2.imshow('liveness_detection',im)
            if cv2.waitKey(1) &0xFF == ord('q'):
                break

            counter_ok_consecutives += 1
            if counter_ok_consecutives == limit_consecutives:
                counter_ok_questions += 1
                counter_try = 0
                counter_ok_consecutives = 0
                break
            else:
                continue

        elif challenge_res == "fail":
            counter_try += 1
            im = show_image(cam,question+" : fail")
            cv2.imshow('liveness_detection',im)
            cv2.waitKey(1)
        elif j == limit_try-1:
            break
    if counter_ok_questions ==  limit_questions:
        user_name = is_exists(frame)
        while True:
            im = show_image(cam,f"LIFENESS SUCCESSFUL - {user_name}",color = (0,255,0))
            cv2.imshow('liveness_detection',im)
            if cv2.waitKey(1) &0xFF == ord('q'):
                break
    elif j == limit_try-1:
        while True:
            im = show_image(cam,"LIFENESS FAIL")
            cv2.imshow('liveness_detection',im)
            if cv2.waitKey(1) &0xFF == ord('q'):
                break
        break 

    else:
        continue

cam.release()
cv2.destroyAllWindows()
