FROM python:3.10-slim

# Instala dependencias del sistema para dlib y opencv
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgtk2.0-dev \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean

# Crea el directorio de trabajo
WORKDIR /app

# Copia los archivos del proyecto
COPY . /app

# Instala las dependencias
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expone el puerto 5000
EXPOSE 5000

# Comando para iniciar la app
CMD ["python", "app.py"]
