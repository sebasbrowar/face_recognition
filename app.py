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
        # Cambiar para aceptar FormData
        if 'frame' not in request.files:
            return jsonify({"error": "No se proporcionó imagen"}), 400

        file = request.files['frame']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Invertir horizontalmente la imagen (como se ve en pantalla)
        img = cv2.flip(img, 1)

        if img is None:
            return jsonify({"error": "Imagen corrupta"}), 400

        # Optimización: Redimensionar
        h, w = img.shape[:2]
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ubicaciones = face_recognition.face_locations(rgb, model="hog")

        if not ubicaciones:
            return jsonify({
                "success": True,
                "count": 0,
                "locations": [],
                "names": []
            })

        codigos = face_recognition.face_encodings(rgb, ubicaciones)

        nombres = []
        for cod in codigos:
            distancias, indices = kdtree.query(cod, k=1)  # Solo el mejor match

            # Umbral más permisivo (0.7)
            if distancias < 0.7:
                nombres.append(nombres_rostros[indices])
            else:
                nombres.append("Desconocido")

        return jsonify({
            "success": True,
            "count": len(ubicaciones),
            "locations": ubicaciones,
            "names": nombres
        })

    except Exception as e:
        logging.error(f"Error en /reconocer: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/models/<path:filename>")
def serve_model(filename):
    return send_from_directory("models", filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
