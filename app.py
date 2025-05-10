import pickle
import face_recognition
import numpy as np
from flask import Flask, request, jsonify, render_template
import cv2
import os
from PIL import Image
from io import BytesIO

# Cargar embeddings
with open('rostros.pkl', 'rb') as f:
    known_faces = pickle.load(f)

known_encodings = [encoding for name, encoding in known_faces]
known_names = [name for name, encoding in known_faces]

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reconocer', methods=['POST'])
def reconocer():
    if 'frame' not in request.files:
        return jsonify({'nombre': 'Sin imagen'}), 400

    file = request.files['frame']
    img = Image.open(BytesIO(file.read()))
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Desconocido"
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
        return jsonify({'nombre': name})

    return jsonify({'nombre': 'Sin rostro'})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
