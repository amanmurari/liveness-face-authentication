import cv2
import mediapipe as mp
import urllib.request
import os
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from scipy.spatial import distance as dist
import numpy as np
from configs import *
if not os.path.exists(MODEL_PATH):
    print("Downloading face landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

COUNTER=0
BLINK_THRESHOLD=0.25
EYE_AR_CONSEC_FRAMES = 5
TOTAL=0

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.5,
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def get_eye_coordinates(landmarks, eye_indices):
    eye = []
    for idx in eye_indices:
        lm = landmarks[idx]
        eye.append((lm.x, lm.y))
    return np.array(eye)


def draw_face_landmarks(image, detection_result):
    annotated = image.copy()
    h, w = annotated.shape[:2]
    for face_landmarks in detection_result.face_landmarks:
        for idx in LEFT_EYE + RIGHT_EYE:
            lm = face_landmarks[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (cx, cy), 2, (0, 255, 0), -1)
    return annotated


def process_blinks(annotated_image,counter,total):
    global COUNTER, TOTAL
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    results = face_landmarker.detect(mp_image)
    if not results.face_landmarks:
        return counter, total

    
    left_eye = get_eye_coordinates(results.face_landmarks[0], LEFT_EYE)
    right_eye = get_eye_coordinates(results.face_landmarks[0], RIGHT_EYE)
    ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

    if ear < BLINK_THRESHOLD:
        counter += 1
        if counter >= EYE_AR_CONSEC_FRAMES:
            total += 1
            counter = 0
    else:
        counter = 0

    return counter, total





if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        annotated_image = frame.copy()



        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        results = face_landmarker.detect(mp_image)
        if results.face_landmarks:
            annotated_image = draw_face_landmarks(annotated_image, results)
            left_eye = get_eye_coordinates(results.face_landmarks[0], LEFT_EYE)
            right_eye = get_eye_coordinates(results.face_landmarks[0], RIGHT_EYE)
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear=(left_ear + right_ear) / 2.0
            if ear < BLINK_THRESHOLD:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                    COUNTER=0
                    cv2.putText(annotated_image, "Blink Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                COUNTER = 0
        annotated_image = cv2.flip(annotated_image, 1)
        cv2.putText(annotated_image, f"Blinks: {TOTAL}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("frame", annotated_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
