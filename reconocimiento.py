import face_recognition
import cv2
import pickle

# Cargar rostros codificados y nombres desde archivo
with open("rostros.pkl", "rb") as f:
    rostros_codificados, nombres_rostros = pickle.load(f)

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    pequeño_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_pequeño = cv2.cvtColor(pequeño_frame, cv2.COLOR_BGR2RGB)

    ubicaciones = face_recognition.face_locations(rgb_pequeño)
    codigos = face_recognition.face_encodings(rgb_pequeño, ubicaciones)

    for cod, ubi in zip(codigos, ubicaciones):
        coincidencias = face_recognition.compare_faces(rostros_codificados, cod, tolerance=0.5)
        nombre = "Desconocido"

        if True in coincidencias:
            index = coincidencias.index(True)
            nombre = nombres_rostros[index]

        top, right, bottom, left = [v * 4 for v in ubi]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, nombre, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Reconocimiento facial', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
