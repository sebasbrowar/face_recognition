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
    return render_template("index_v0.html")

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

    for cod in codigos:
        coincidencias = face_recognition.compare_faces(rostros_codificados, cod)
        if True in coincidencias:
            index = coincidencias.index(True)
            return jsonify({"nombre": nombres_rostros[index]})

    return jsonify({"nombre": "Desconocido"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

