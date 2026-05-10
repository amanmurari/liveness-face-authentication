import sqlite3
import face_recognition as fc
import pickle
import threading
import os

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'face.db')
db = sqlite3.connect(_DB_PATH, check_same_thread=False)
cursor = db.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY, encoding BLOB)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)''')
db.commit()

_lock = threading.Lock()


def insert_face(name, face_encoding):
    with _lock:
        cursor.execute('''INSERT INTO users (name) VALUES (?)''', (name,))
        user_id = cursor.lastrowid
        cursor.execute('''INSERT INTO faces (id, encoding) VALUES (?, ?)''', (user_id, pickle.dumps(face_encoding)))
        db.commit()

def load_faces():
    with _lock:
        cursor.execute('''SELECT users.id, faces.encoding FROM faces JOIN users ON faces.id = users.id''')
        rows = cursor.fetchall()
    known_face_id = []
    known_face_encodings = []
    for id, encoding in rows:
        known_face_id.append(id)
        known_face_encodings.append(pickle.loads(encoding))
    return known_face_id, known_face_encodings

def get_user_name(user_id):
    with _lock:
        cursor.execute('''SELECT name FROM users WHERE id = ?''', (user_id,))
        result = cursor.fetchone()
    return result[0] if result else None


if __name__ == "__main__":
    name="aman"
    image=fc.load_image_file('images/linddin.jpeg')
    face_encoding=fc.face_encodings(image)[0]
    insert_face(name,face_encoding)
    print(load_faces())  