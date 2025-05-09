import face_recognition
import os
import numpy as np
import pickle
import cv2

RUTA_CONOCIDOS = 'C:/Users/sbrxb/PycharmProjects/face_recognition/conocidos'
rostros_codificados = []
nombres_rostros = []

print("\n🧠 Procesando rostros...")
for persona in os.listdir(RUTA_CONOCIDOS):
    ruta_persona = os.path.join(RUTA_CONOCIDOS, persona)
    if not os.path.isdir(ruta_persona):
        continue

    for archivo in os.listdir(ruta_persona):
        if not archivo.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        ruta_img = os.path.join(ruta_persona, archivo)

        try:
            img_bgr = cv2.imread(ruta_img)

            if img_bgr is None:
                print(f"❌ No se pudo leer (imagen corrupta o vacía): {ruta_img}")
                continue

            if img_bgr.dtype != np.uint8:
                print(f"❌ Tipo de imagen no compatible (no uint8): {ruta_img}")
                continue

            if len(img_bgr.shape) != 3 or img_bgr.shape[2] != 3:
                print(f"❌ Imagen no RGB de 3 canales: {ruta_img}")
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            codigos = face_recognition.face_encodings(img_rgb)

            if codigos:
                rostros_codificados.append(codigos[0])
                nombres_rostros.append(persona)
                print(f"✔️ Rostro detectado: {archivo} ({persona})")
            else:
                print(f"😕 No se detectó rostro en: {archivo}")

        except Exception as e:
            print(f"❌ Error procesando {archivo}: {e}")

# Guardar codificaciones
if rostros_codificados:
    with open("rostros.pkl", "wb") as f:
        pickle.dump((rostros_codificados, nombres_rostros), f)
    print(f"\n✅ Se codificaron {len(rostros_codificados)} rostros.")
    print("💾 Guardado en 'rostros.pkl'")
else:
    print("\n⚠️ No se codificó ningún rostro.")
