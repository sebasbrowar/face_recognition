const video = document.getElementById("video");
const resultado = document.getElementById("resultado");

// Activar cámara
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    });

function capturar() {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    const imagenData = canvas.toDataURL("image/jpeg");

    fetch("/reconocer", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ imagen: imagenData })
    })
    .then(response => response.json())
    .then(data => {
        if (data.nombre) {
            resultado.textContent = `👤 Rostro: ${data.nombre}`;
        } else {
            resultado.textContent = `❌ No se reconoció ningún rostro.`;
        }
    })
    .catch(err => {
        resultado.textContent = `⚠️ Error: ${err}`;
    });
}
