# FaceGuard — Liveness Detection

A web-based face liveness detection system. It asks the user to perform a few random actions (blink, smile, turn head) before verifying their identity against a registered face. Runs locally in the browser using your webcam.

---

## What it does

1. You register your face by capturing a photo and entering your name.
2. During verification, the system shows a random challenge (e.g. "BLINK EYES", "HAPPY", "TURN FACE LEFT").
3. You need to pass 3 of these challenges to proceed.
4. After that it checks if the face matches anyone in the database and shows the result.

The liveness step makes sure it's a real person in front of the camera, not a photo or video.

---

## Requirements

- Python 3.9 or 3.10 (face_recognition has build issues on 3.11+)
- A working webcam
- Windows / Linux / macOS

---

## Installation

**1. Clone or download the project**

```
git clone <repo-url>
cd liveness
```

**2. Create a virtual environment**

```
python -m venv venv
```

Activate it:

- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

**3. Install dependencies**

```
pip install flask opencv-python face-recognition mediapipe torch torchvision scipy numpy
```

> `face-recognition` needs `cmake` and `dlib` to build. On Windows, install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first. On Ubuntu run `sudo apt install cmake libboost-all-dev` before pip install.

**4. Download the MediaPipe face landmark model**

The app downloads it automatically on first run. If you're offline, grab `face_landmarker.task` from the MediaPipe model card and drop it in the project root.

---

## Running

```
python app.py
```

Then open your browser at `http://localhost:5000`.

Flask runs in debug mode by default so it will reload on code changes.

---

## How to use

**Register a face**

- Go to the left panel.
- Click "Capture" to take a photo (make sure your face is clearly visible and well-lit).
- Type your name and click "Register".
- You'll see a confirmation message. You can register multiple people.

**Run verification**

- Go to the right panel.
- Click "Start".
- Follow the challenge shown on screen. The box turns green when you're doing it right.
- After passing all challenges, it will tell you who it found (or "face not in database" if you haven't registered).

---

## Project structure

```
liveness/
    app.py               Flask app and API routes
    liveness.py          Core liveness + face recognition logic
    eyeblinks.py         Blink detection via MediaPipe
    profiling.py         Head direction detection via OpenCV cascades
    emotions.py          Emotion detection (random mock — plug in your model here)
    questions.py         Challenge question bank
    configs.py           Shared config values
    dbops/
        insertation.py   SQLite helpers for storing/loading face encodings
    templates/
        index.html       The frontend
    face.db              Auto-created on first run
    face_landmarker.task Auto-downloaded on first run
```

---

## Notes

- The emotion model (`emotions.py`) is currently mocked — it returns a random class. To use a real model, uncomment the model loading code and point it to your weights file at `weights/emotions.pth`.
- Face encodings are stored in `face.db` as serialized numpy arrays. If you want to start fresh, just delete the file.
- The app uses a single SQLite connection shared across threads. It works fine for local use but isn't meant for production.
