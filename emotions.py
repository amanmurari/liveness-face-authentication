import torch
from src.spoofingm import SpoofNN
from torchvision import transforms
import cv2
import numpy as np,random

classes = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprise']

tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# model = SpoofNN()
# model.load_state_dict(torch.load('weights/emotions.pth', map_location=torch.device('cpu')))
# model.eval()

@torch.inference_mode()
def get_emotion(frame):

    return random.choice(classes)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # face_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    # if len(face_rects) == 0:
    #     return None
    # emotion = None
    # for (x, y, w, h) in face_rects:
    #     face_roi = gray[y:y+h, x:x+w]
    #     tensor = tf(face_roi).unsqueeze(0)
    #     output = model(tensor)
    #     emotion = classes[torch.argmax(output[0]).item()]
        
    # return emotion
