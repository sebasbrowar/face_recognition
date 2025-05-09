from flask import Flask, render_template, request, jsonify
import face_recognition
import numpy as np
import pickle
import cv2
import base64

app = Flask(__name__)

# Cargar rostros conocidos
with open("rostros.pkl", "rb") as f:
    rostros_codificados, nombres_rostros = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/reconocer", methods=["POST"])
def reconocer():
    data_url = request.json.get("imagen")
    if not data_url:
        return jsonify({"error": "No se recibió imagen"}), 400

    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    pequeño = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    rgb = cv2.cvtColor(pequeño, cv2.COLOR_BGR2RGB)

    ubicaciones = face_recognition.face_locations(rgb)
    codigos = face_recognition.face_encodings(rgb, ubicaciones)

    for ubicacion, cod in zip(ubicaciones, codigos):
        coincidencias = face_recognition.compare_faces(rostros_codificados, cod)
        nombre = "Desconocido"

        if True in coincidencias:
            index = coincidencias.index(True)
            nombre = nombres_rostros[index]

        top, right, bottom, left = [v * 4 for v in ubicacion]
        cv2.rectangle(img, (left, top), (right, bottom), (255, 140, 0), 2)
        cv2.putText(img, nombre, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 140, 0), 2)

    _, buffer = cv2.imencode(".jpg", img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    return jsonify({"imagen_procesada": "data:image/jpeg;base64," + img_base64})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

