import cv2
import numpy as np

_detect_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
_detect_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')


def _detect(img, cascade):
    rects, _, confidence = cascade.detectMultiScale3(
        img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE, outputRejectLevels=True)
    if len(rects) == 0:
        return (), ()
    rects[:, 2:] += rects[:, :2]
    return rects, confidence


def _convert_rightbox(img, box_right):
    res = np.array([])
    _, x_max = img.shape
    for box_ in box_right:
        box = np.copy(box_)
        box[0] = x_max - box_[2]
        box[2] = x_max - box_[0]
        res = np.expand_dims(box, axis=0) if res.size == 0 else np.vstack((res, box))
    return res


def get_face_direction(gray):
    """Returns 'frontal', 'left', 'right', or None based on detected face orientation."""
    box_frontal, _ = _detect(gray, _detect_frontal)
    if len(box_frontal) > 0:
        return "frontal"

    box_left, _ = _detect(gray, _detect_profile)
    if len(box_left) > 0:
        return "left"

    gray_flipped = cv2.flip(gray, 1)
    box_right, _ = _detect(gray_flipped, _detect_profile)
    if len(box_right) > 0:
        return "right"

    return None


def draw_detections(img, gray):
    """Draws bounding boxes and direction labels on img. Returns annotated image."""
    boxes, names = [], []

    box_frontal, _ = _detect(gray, _detect_frontal)
    if len(box_frontal) > 0:
        boxes += list(box_frontal)
        names += ["frontal"] * len(box_frontal)

    box_left, _ = _detect(gray, _detect_profile)
    if len(box_left) > 0:
        boxes += list(box_left)
        names += ["left"] * len(box_left)

    gray_flipped = cv2.flip(gray, 1)
    box_right, _ = _detect(gray_flipped, _detect_profile)
    if len(box_right) > 0:
        box_right = _convert_rightbox(gray, box_right)
        boxes += list(box_right)
        names += ["right"] * len(box_right)

    annotated = img.copy()
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 3)
        cv2.putText(annotated, names[i], (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return annotated


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.flip(frame, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        direction = get_face_direction(gray)
        print(f"Direction: {direction}")

        result = draw_detections(img, gray)
        cv2.imshow("profile_detection", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
