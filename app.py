import pickle
import face_recognition
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
import cv2, os
from io import BytesIO
from PIL import Image

# Cargar embeddings desde el archivo 'rostros.pkl'
with open('rostros.pkl', 'rb') as f:
    known_faces = pickle.load(f)

# Desempaquetar los encodings de cada persona
known_encodings = []
known_names = []

for i, encodings in enumerate(known_faces[:-1]):  # Excluimos el último elemento que es la lista de nombres
    known_encodings.extend(encodings)
    known_names.extend([known_faces[-1][i]] * len(encodings))  # Añadimos el nombre correspondiente a cada encoding

app = Flask(__name__, static_folder='static', template_folder='templates')


# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')


# Ruta para el video feed (streaming en vivo desde la cámara)
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Ruta para el reconocimiento de la cara cuando se envía una imagen desde el cliente
@app.route('/reconocer', methods=['POST'])
def reconocer():
    file = request.files['image']  # Recibe la imagen desde el cliente
    img = Image.open(file.stream)  # Convierte la imagen recibida en un objeto PIL
    frame = np.array(img)  # Convierte la imagen PIL a un array de NumPy (compatible con OpenCV)

    rgb_frame = frame[:, :, ::-1]  # Convierte la imagen a RGB (para face_recognition)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Desconocido"

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
        names.append(name)

    # Devolver el resultado (por ejemplo, el nombre de la persona reconocida)
    return jsonify({'names': names, 'locations': face_locations})


def gen_frames():
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break

        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Desconocido"

            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
            names.append(name)

        # Dibujar rectángulos y nombres
        for (top, right, bottom, left), name in zip(face_locations, names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
