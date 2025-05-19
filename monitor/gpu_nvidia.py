#!/usr/bin/env python3
"""
pip install pynvml pandas
gpu_monitor.py  ▸  Registro continuo de sensores GPU → CSV
Ejemplo de uso: W
    start /b python gpu_nvidia.py -o gpu_stats.csv -i 0.5
    # …lanzar tu entrenamiento en otro terminal…
    # Ctrl-C para detener la captura
Ejemplo de uso: Lx
nohup python3 gpu_nvidia.py -o gpu_stats.csv -i 0.5 > gpu_monitor.log 2>&1 &
Otra opción:
nvidia-smi \
  --query-gpu=timestamp,name,index,utilization.gpu,\
memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv,noheader,nounits \
  -lms 500   >> gpu_stats.csv

"""
"""
gpu_monitor.py  ▸  Registro continuo de sensores GPU → CSV
Requisitos: pip install pynvml pandas
"""

import time, csv, argparse
import datetime as dt
from datetime import timezone
from pynvml import (
    nvmlInit, nvmlShutdown, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName, nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetTemperature, NVML_TEMPERATURE_GPU,
    nvmlDeviceGetPowerUsage, nvmlDeviceGetMemoryInfo
)

def snapshot(dev):
    """Devuelve una lista con las métricas clave de una GPU."""
    # nvmlDeviceGetName ya devuelve str en versiones recientes
    name  = nvmlDeviceGetName(dev)
    util  = nvmlDeviceGetUtilizationRates(dev)
    mem   = nvmlDeviceGetMemoryInfo(dev)
    temp  = nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU)
    power = nvmlDeviceGetPowerUsage(dev) / 1000.0   # mW → W
    return [name,
            util.gpu, util.memory,
            mem.used / 1e6, mem.total / 1e6,
            temp, power]

def main(out_path: str, interval: float):
    nvmlInit()
    n_gpus = nvmlDeviceGetCount()
    handles = [nvmlDeviceGetHandleByIndex(i) for i in range(n_gpus)]

    header = ["timestamp_utc", "gpu_idx", "gpu_name",
              "gpu_util_%", "mem_util_%", "mem_used_MB",
              "mem_total_MB", "temp_C", "power_W"]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        try:
            while True:
                # timestamp timezone-aware
                tstamp = dt.datetime.now(timezone.utc).isoformat()
                for idx, h in enumerate(handles):
                    row = [tstamp, idx] + snapshot(h)
                    writer.writerow(row)
                f.flush()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n⏹ Monitorización detenida por el usuario.")
    nvmlShutdown()

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Guarda métricas de sensores GPU NVIDIA en un CSV.")
    p.add_argument("-o", "--output",  default="gpu_stats.csv",
                   help="Ruta y nombre del archivo CSV")
    p.add_argument("-i", "--interval", type=float, default=1.0,
                   help="Intervalo de muestreo en segundos")
    args = p.parse_args()
    main(args.output, args.interval)

