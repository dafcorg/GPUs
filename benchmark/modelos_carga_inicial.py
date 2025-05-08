import torch
import torchvision.models as models
from transformers import AutoModel, AutoTokenizer
import os
import time
import threading
import pandas as pd
import datetime
import tkinter as tk
from tkinter import messagebox, ttk

# Intentar importar pynvml para monitoreo de GPU
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates,
                         nvmlDeviceGetMemoryInfo, nvmlShutdown, nvmlDeviceGetPowerUsage
    nvmlInit()
    gpu_available = True
    handle = nvmlDeviceGetHandleByIndex(0)
except:
    gpu_available = False

# Configuración de modelos disponibles
modelos = {
    "ResNet (imagen)": ("torchvision", "resnet50"),
    "GPT-2 (texto)": ("huggingface", "gpt2"),
    "T5 (traducción/texto)": ("huggingface", "t5-small"),
    "Wav2Vec2.0 (audio)": ("huggingface", "facebook/wav2vec2-base-960h"),
    "Whisper (audio a texto)": ("huggingface", "openai/whisper-base"),
    "YOLOv5 (detección de objetos)": ("torch.hub", "ultralytics/yolov5"),
    "LLaMA 2 7B": ("huggingface", "meta-llama/Llama-2-7b-hf")
}

# Carpeta de resultados
os.makedirs("results", exist_ok=True)

def monitorear(dispositivo, stop_event, stats_list):
    while not stop_event.is_set():
        if dispositivo == 'gpu' and gpu_available:
            util = nvmlDeviceGetUtilizationRates(handle)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            watts = nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
            stats = {
                'timestamp': time.time(),
                'gpu_usage_percent': util.gpu,
                'vram_used_mb': mem_info.used / 1024**2,
                'power_watts': watts
            }
        else:
            stats = {
                'timestamp': time.time(),
                'gpu_usage_percent': 0,
                'vram_used_mb': 0,
                'power_watts': 0
            }
        stats_list.append(stats)
        time.sleep(1)

def prueba_esfuerzo(nombre_modelo, tipo, id_modelo, dispositivo):
    stats_list = []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitorear, args=(dispositivo, stop_event, stats_list))
    monitor_thread.start()

    start_time = time.time()
    device = torch.device("cuda" if dispositivo == 'gpu' and torch.cuda.is_available() else "cpu")

    # ===== Carga del modelo =====
    if tipo == "torchvision":
        model = getattr(models, id_modelo)(pretrained=True).to(device).eval()
        data = torch.randn(5000, 3, 224, 224).to(device)
        with torch.no_grad():
            for i in range(data.shape[0]):
                model(data[i:i+1])

    elif tipo == "huggingface":
        tokenizer = AutoTokenizer.from_pretrained(id_modelo, use_auth_token=True)
        model = AutoModel.from_pretrained(id_modelo, use_auth_token=True).to(device).eval()
        prompts = ["Hello, how are you?" for _ in range(1000)]
        with torch.no_grad():
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                model(**inputs)

    elif tipo == "torch.hub":
        model = torch.hub.load(id_modelo, 'yolov5s', pretrained=True).to(device)
        imgs = torch.randn(1000, 3, 640, 640).to(device)
        with torch.no_grad():
            for i in range(imgs.shape[0]):
                model(imgs[i:i+1])

    end_time = time.time()
    stop_event.set()
    monitor_thread.join()

    df = pd.DataFrame(stats_list)
    df['start_time'] = datetime.datetime.fromtimestamp(start_time)
    df['end_time'] = datetime.datetime.fromtimestamp(end_time)
    df['duration_sec'] = end_time - start_time

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    filename = f"results/{nombre_modelo.replace(' ', '_')}_{dispositivo}_{timestamp}.csv"
    df.to_csv(filename, index=False)
    return filename

# ===========================
# Interfaz
# ===========================

def iniciar():
    modelo = modelo_cb.get()
    dispositivo = dispositivo_cb.get().lower()
    if not modelo or not dispositivo:
        messagebox.showerror("Error", "Seleccione un modelo y un dispositivo")
        return

    nombre_modelo, tipo, id_modelo = modelo, *modelos[modelo]
    try:
        archivo = prueba_esfuerzo(nombre_modelo, tipo, id_modelo, dispositivo)
        messagebox.showinfo("Prueba finalizada", f"Resultados guardados en:\n{archivo}")
    except Exception as e:
        messagebox.showerror("Error durante la prueba", str(e))

root = tk.Tk()
root.title("Prueba de Esfuerzo de Modelos IA")
root.geometry("500x250")

frame = ttk.Frame(root, padding=20)
frame.pack(fill="both", expand=True)

ttk.Label(frame, text="Selecciona el modelo:").pack()
modelo_cb = ttk.Combobox(frame, values=list(modelos.keys()), state="readonly")
modelo_cb.pack(pady=5)

ttk.Label(frame, text="Selecciona el dispositivo:").pack()
dispositivo_cb = ttk.Combobox(frame, values=["GPU", "CPU"], state="readonly")
dispositivo_cb.pack(pady=5)

btn = ttk.Button(frame, text="Iniciar prueba de esfuerzo", command=iniciar)
btn.pack(pady=20)

root.mainloop()

# Cerrar NVML si fue iniciado
if gpu_available:
    nvmlShutdown()
