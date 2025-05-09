const video = document.getElementById("video");
const resultado = document.getElementById("resultado");

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => video.srcObject = stream);

function capturar() {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  const dataUrl = canvas.toDataURL("image/jpeg");

  fetch("/reconocer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ imagen: dataUrl })
  })
  .then(res => res.json())
  .then(data => {
    resultado.innerText = "👤 Reconocido: " + data.nombre;
  })
  .catch(() => resultado.innerText = "❌ Error en el reconocimiento");
}
