import uuid
import base64
import numpy as np
import cv2
from random import randint
from flask import Flask, request, jsonify, render_template
import face_recognition as fc
from liveness import detect_liveness, is_exists, reload_faces
from dbops.insertation import insert_face
from questions import question_bank

app = Flask(__name__)
app.secret_key = 'liveness-secret-key'

_sessions = {}

LIMIT_CONSECUTIVES = 3
LIMIT_QUESTIONS = 6
LIMIT_TRY = 50


def _decode_frame(b64: str) -> np.ndarray:
    if ',' in b64:
        b64 = b64.split(',')[1]
    buf = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _new_question():
    """Pick a random question and return fresh per-question state."""
    return {
        'question': question_bank(randint(0, 5)),
        'counter': 0,
        'total': 0,
        'prev_total': 0,
        'ok_consecutives': 0,
        'frame_count': 0,
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    name = (data.get('name') or '').strip()
    image_b64 = data.get('image', '')
    if not name:
        return jsonify({'success': False, 'error': 'Name is required'})
    frame = _decode_frame(image_b64)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = fc.face_encodings(rgb)
    if not encodings:
        return jsonify({'success': False, 'error': 'No face detected in image'})
    insert_face(name, encodings[0])
    reload_faces()
    return jsonify({'success': True})


@app.route('/api/session/start', methods=['POST'])
def start_session():
    session_id = str(uuid.uuid4())
    q_state = _new_question()
    _sessions[session_id] = {
        'q_state': q_state,
        'ok_questions': 0,
        'done': False,
        'success': False,
        'name': None,
    }
    return jsonify({
        'session_id': session_id,
        'challenge': q_state['question'],
        'progress': 0,
        'total_challenges': LIMIT_QUESTIONS,
    })


@app.route('/api/session/frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    session_id = data.get('session_id', '')
    if session_id not in _sessions:
        return jsonify({'error': 'Invalid session'}), 400

    s = _sessions[session_id]
    if s['done']:
        return jsonify({'done': True, 'success': s['success'], 'name': s['name']})

    frame = _decode_frame(data.get('frame', ''))
    q = s['q_state']
    question = q['question']
    q['frame_count'] += 1

    # --- run liveness check (mirrors spoofing.py logic) ---
    if question == 'blink eyes':
        q['counter'], q['total'] = detect_liveness(frame, q['total'], q['counter'], question)
        passed = q['total'] > q['prev_total']
    else:
        passed = detect_liveness(frame, q['total'], q['counter'], question)

    # --- evaluate result ---
    if passed:
        challenge_res = 'pass'
    else:
        challenge_res = 'fail'

    if challenge_res == 'pass':
        q['ok_consecutives'] += 1
        if q['ok_consecutives'] == LIMIT_CONSECUTIVES:
            s['ok_questions'] += 1

            if s['ok_questions'] == LIMIT_QUESTIONS:
                user_name = is_exists(frame)
                s['done'] = True
                s['success'] = True
                s['name'] = user_name
                return jsonify({'done': True, 'success': True, 'name': user_name})

            s['q_state'] = _new_question()
            return jsonify({
                'done': False,
                'status': 'challenge_passed',
                'challenge': s['q_state']['question'],
                'progress': s['ok_questions'],
                'total_challenges': LIMIT_QUESTIONS,
            })

    elif challenge_res == 'fail':
        # do NOT reset ok_consecutives on fail — matches spoofing.py behaviour
        # (need 3 total passes per question, not 3 strictly consecutive)
        if q['frame_count'] >= LIMIT_TRY:
            s['done'] = True
            s['success'] = False
            return jsonify({'done': True, 'success': False, 'name': None})

    return jsonify({
        'done': False,
        'status': challenge_res,
        'challenge': question,
        'progress': s['ok_questions'],
        'total_challenges': LIMIT_QUESTIONS,
    })


if __name__ == '__main__':
    app.run(debug=True)
