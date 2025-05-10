# 📸 Reconocimiento Facial en Tiempo Real

Este proyecto permite detectar y reconocer rostros en tiempo real utilizando la librería `face_recognition` en Python 3.10.9. Ha sido desplegado en la nube mediante [Render](https://render.com), lo que permite acceder a la aplicación desde cualquier dispositivo con navegador.

## 🗂 Estructura del Proyecto (GitHub)

La organización del repositorio [sebasbrowar/face_recognition](https://github.com/sebasbrowar/face_recognition) sigue esta estructura:

```plaintext
face_recognition/
├── app.py                       # Aplicación principal Flask
├── requirements.txt            # Dependencias de Python
├── runtime.txt                 # Versión de Python para despliegue (ej. python-3.10.9)
├── rostros.pkl                 # Base de datos de rostros codificados
├── .gitignore                  # Archivos a ignorar por Git
├── conocidos/                  # Directorio de imágenes para entrenamiento
│   ├── persona1/               # Ejemplo de estructura por persona
│   │   ├── foto1.jpg
│   │   └── foto2.jpg
│   └── persona2/
│       └── retrato.jpg
├── static/                     # Archivos estáticos
│   └── js/
│       └── face-api.min.js     # Biblioteca para procesamiento facial
├── templates/                  # Plantillas HTML
│   └── index.html              # Página principal
├── models/                     # Modelos pre-entrenados
│   ├── face_landmark_68_model-weights_manifest.json
│   ├── face_recognition_model-weights_manifest.json
│   └── tiny_face_detector_model-weights_manifest.json
├── generar_rostros.py          # Script para generar rostros.pkl
├── Dockerfile                  # Configuración para contenedores
└── README.md                   # Documentación del proyecto
```

---

## 🔧 Funcionalidad

- Detecta y reconoce rostros conocidos desde la cámara del dispositivo.
- Permite registrar nuevos rostros y generar el archivo `rostros.pkl`.
- Backend en Python (Flask o FastAPI).
- Frontend en HTML + JavaScript, capturando video con `getUserMedia()`.
- Utiliza modelos preentrenados para detección facial:
  - `face_landmark_68_model-weights_manifest.json`
  - `face_recognition_model-weights_manifest.json`
  - `tiny_face_detector_model-weights_manifest.json`
  *(Descargados de [face-api.js-models](https://github.com/justadudewhohacks/face-api.js-models/tree/master))*
- **static/js/face-api.min.js**: Biblioteca esencial para el procesamiento facial en el frontend, descargada del mismo repositorio que los modelos.

---

## 🌐 Despliegue en Render

La aplicación se ha desplegado en [Render](https://render.com), lo cual permite su acceso vía navegador.

### ❗ Consideraciones desde el celular

- El reconocimiento funciona, pero la cámara puede no mostrarse por:
  - Limitaciones del navegador móvil (permiso de cámara), al darle permiso debería funcionar.
  - Render puede tener restricciones para acceder directamente a dispositivos.
  - Problemas de compatibilidad con `getUserMedia()`.

✅ Aun sin ver la imagen, los frames se siguen enviando y el reconocimiento se realiza correctamente.

---

## 🧠 Backend y Frontend

### 🔙 Backend

- Python 3.10.9 con `face_recognition`, `OpenCV` y `Flask`
- Carga rostros codificados desde `rostros.pkl`
- Procesa imágenes y devuelve nombres reconocidos
- Utiliza modelos preentrenados para detección precisa

### 🎨 Frontend

- Captura video con `getUserMedia()`
- Envía frames al backend con `fetch`
- Muestra resultados en tiempo real
- Usa `face-api.min.js` para procesamiento inicial

---

## 🧪 Generar `rostros.pkl` con tus imágenes

Para que el sistema reconozca personas, primero debes crear el archivo `rostros.pkl` con las codificaciones de los rostros conocidos.

### 📁 Estructura esperada

Organiza tus imágenes así:
```plaintext
conocidos/
├── persona1/
│ ├── cara1.jpg
│ └── cara2.png
├── persona2/
│ └── selfie.jpeg
```

Cada subcarpeta representa una persona y contiene imágenes de su rostro.
El archivo rostros.pkl debe colocarse en el mismo directorio donde se ejecuta el backend (junto a app.py, por ejemplo).

---

## 📦 Instalación local

### 1. Clonar repositorio

```bash
git clone https://github.com/sebasbrowar/face_recognition.git
cd face_recognition
```

### 2. Entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar
```bash
python app.py
```

Abre tu navegador en http://localhost:5000

---

## 🐳 Docker (opcional)

El Dockerfile viene en el proyecto de GitHub.

### Construir y correr
```bash
docker build -t face-recognition .
docker run -p 5000:5000 face-recognition
```