from flask import Flask, render_template, request, jsonify, send_from_directory
import face_recognition
import numpy as np
import pickle
import cv2
import base64
import os
from scipy.spatial import KDTree
import logging

app = Flask(__name__, static_folder='static', template_folder='templates')
logging.basicConfig(level=logging.INFO)

# Variables globales para almacenar los datos de rostros
rostros_codificados = []
nombres_rostros = []
kdtree = None


def cargar_rostros():
    global rostros_codificados, nombres_rostros, kdtree

    try:
        if not os.path.exists("rostros.pkl"):
            raise FileNotFoundError("Archivo 'rostros.pkl' no encontrado.")

        with open("rostros.pkl", "rb") as f:
            rostros_codificados, nombres_rostros = pickle.load(f)

        if not rostros_codificados or not nombres_rostros:
            raise ValueError("Datos en 'rostros.pkl' están vacíos o corruptos.")

        # Convertir a numpy array y construir KDTree
        rostros_array = np.array(rostros_codificados)
        kdtree = KDTree(rostros_array)
        logging.info(f"Se cargaron {len(rostros_codificados)} rostros. KDTree construido.")

    except Exception as e:
        logging.error(f"Error al cargar rostros: {str(e)}")
        raise RuntimeError(f"No se pudo iniciar el servicio: {str(e)}")


# Cargar rostros al iniciar la aplicación
cargar_rostros()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/reconocer", methods=["POST"])
def reconocer():
    try:
        data_url = request.json.get("imagen", "")
        if not data_url or len(data_url) > 10_000_000:
            return jsonify({"error": "Datos de imagen no válidos o demasiado grandes"}), 400

        # Decodificar imagen
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Imagen corrupta o formato no soportado"}), 400

        # Optimización: Redimensionar si es muy grande
        h, w = img.shape[:2]
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        # Convertir a RGB y detectar rostros
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ubicaciones = face_recognition.face_locations(rgb, model="hog")

        if not ubicaciones:
            return jsonify({"error": "No se detectaron rostros"}), 400

        codigos = face_recognition.face_encodings(rgb, ubicaciones)
        logging.info(f"Caras detectadas: {len(codigos)}")

        # Buscar coincidencias usando KDTree
        resultados = []
        for cod in codigos:
            distancias, indices = kdtree.query(cod, k=5)  # Buscar 5 vecinos más cercanos

            # Verificar si la mejor coincidencia está dentro del umbral
            if distancias[0] < 0.6:
                nombre = nombres_rostros[indices[0]]
                resultados.append(nombre)
                logging.info(f"Match encontrado: {nombre} (Distancia: {distancias[0]:.4f})")
            else:
                logging.info(f"Rostro desconocido (Distancia mínima: {distancias[0]:.4f})")

        if resultados:
            return jsonify({"nombre": resultados[0]})  # Devuelve el primer reconocimiento
        else:
            return jsonify({"nombre": "Desconocido"})

    except Exception as e:
        logging.error(f"Error en /reconocer: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error al procesar la imagen: {str(e)}"}), 500


@app.route("/models/<path:filename>")
def serve_model(filename):
    return send_from_directory("models", filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)