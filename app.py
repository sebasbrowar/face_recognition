from flask import Flask, render_template, request, jsonify, send_from_directory
import face_recognition
import numpy as np
import pickle
import cv2
import base64
import os
import logging

app = Flask(__name__, static_folder='static', template_folder='templates')
logging.basicConfig(level=logging.INFO)

# Cargar rostros conocidos con verificación
try:
    if not os.path.exists("rostros.pkl"):
        raise FileNotFoundError("Archivo 'rostros.pkl' no encontrado.")

    with open("rostros.pkl", "rb") as f:
        rostros_codificados, nombres_rostros = pickle.load(f)

    if not rostros_codificados or not nombres_rostros:
        raise ValueError("Datos en 'rostros.pkl' están vacíos o corruptos.")
except Exception as e:
    logging.error(f"Error al cargar rostros: {str(e)}")
    raise RuntimeError(f"No se pudo iniciar el servicio: {str(e)}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/reconocer", methods=["POST"])
def reconocer():
    data_url = request.json.get("imagen", "")
    if not data_url or len(data_url) > 10_000_000:
        return jsonify({"error": "Datos de imagen no válidos o demasiado grandes"}), 400

    try:
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Imagen corrupta o formato no soportado"}), 400

        # Redimensionar si es muy grande (optimización)
        h, w = img.shape[:2]
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ubicaciones = face_recognition.face_locations(rgb, model="hog")
        codigos = face_recognition.face_encodings(rgb, ubicaciones)

        for cod in codigos:
            distancias = face_recognition.face_distance(rostros_codificados, cod)
            best_match_idx = np.argmin(distancias)

            if distancias[best_match_idx] < 0.6:  # Umbral de confianza
                return jsonify({"nombre": nombres_rostros[best_match_idx]})

        return jsonify({"nombre": "Desconocido"})

    except Exception as e:
        logging.error(f"Error en /reconocer: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error al procesar la imagen: {str(e)}"}), 500


@app.route("/models/<path:filename>")
def serve_model(filename):
    return send_from_directory("models", filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)  # Debug=False en producción