from flask import Flask, render_template, request, jsonify
import face_recognition
import cv2
import numpy as np
import pickle
import base64

app = Flask(__name__)

# Cargar los rostros codificados
with open("rostros.pkl", "rb") as f:
    rostros_codificados, nombres_rostros = pickle.load(f)

@app.route("/")
def index():
    return render_template("index_v0.html")

@app.route("/reconocer", methods=["POST"])
def reconocer():
    data = request.get_json()
    imagen_b64 = data.get("imagen")

    if not imagen_b64:
        return jsonify({"error": "No se recibió imagen"}), 400

    try:
        # Eliminar encabezado data:image/jpeg;base64,
        img_data = base64.b64decode(imagen_b64.split(",")[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ubicaciones = face_recognition.face_locations(img_rgb)
        codigos = face_recognition.face_encodings(img_rgb, ubicaciones)

        for cod in codigos:
            coincidencias = face_recognition.compare_faces(rostros_codificados, cod, tolerance=0.5)
            if True in coincidencias:
                index = coincidencias.index(True)
                return jsonify({"nombre": nombres_rostros[index]})

        return jsonify({"nombre": "Desconocido"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
