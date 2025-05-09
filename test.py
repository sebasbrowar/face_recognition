import face_recognition
import cv2

# Ruta de prueba
ruta_img = "C:/Users/sbrxb/PycharmProjects/face_recognition/conocidos/sebastian/frame_300.jpg"

# Leer imagen con OpenCV
img_bgr = cv2.imread(ruta_img)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Obtener codificación
try:
    codigos = face_recognition.face_encodings(img_rgb)
    if codigos:
        print("✅ Rostro codificado exitosamente.")
    else:
        print("⚠️ No se detectó ningún rostro en la imagen.")
except Exception as e:
    print(f"❌ Error al codificar el rostro: {e}")
