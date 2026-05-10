from eyeblinks import process_blinks
from profiling import get_face_direction
from emotions import get_emotion
import cv2
import face_recognition as fc
from dbops.insertation import load_faces, get_user_name
import numpy as np


FACE_ID, FACE_ENCD = load_faces()

def reload_faces():
    global FACE_ID, FACE_ENCD
    FACE_ID, FACE_ENCD = load_faces()

def detect_liveness(frame,total,counter,task):
    if task == 'blink eyes':
        counter,total = process_blinks(frame,counter,total)
        return counter,total

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if task in ['happy','smile','surprise','angry']:
        out_model = get_emotion(gray)
        return out_model==task

    elif task in ['turn face right','turn face left']:
        out_model = get_face_direction(gray)
        if task == 'turn face right':
            return out_model == 'right'
        elif task == 'turn face left':
            return out_model == 'left'

    return False
    


def is_exists(frame):
    if not FACE_ENCD:
        return None
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = fc.face_locations(img)
    encodings = fc.face_encodings(img, face_locations) if face_locations else fc.face_encodings(img)
    for fe in encodings:
        matches = fc.face_distance(FACE_ENCD, fe)
        idx = np.argmin(matches)
        if matches[idx] < 0.65:
            user_id = FACE_ID[idx]
            return get_user_name(user_id)
    return None
