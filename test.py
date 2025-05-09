import pickle
import os
import logging

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

# Añade esto ANTES de cargar el archivo en tu código Flask
print(f"Tamaño del archivo rostros.pkl: {os.path.getsize('rostros.pkl')} bytes")
with open("rostros.pkl", "rb") as f:
    data = pickle.load(f)
    print(f"Número de rostros cargados: {len(data[0])}")  # Debe ser > 0
    print(f"Ejemplo de nombre: {data[1][0]}")  # Muestra el primer nombre
