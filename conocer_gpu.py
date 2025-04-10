import torch

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Número de GPUs disponibles: {num_gpus}")

    for i in range(num_gpus):
        print(f"\n--- GPU {i} ---")
        props = torch.cuda.get_device_properties(i)
        print(f"Nombre: {props.name}")
        print(f"Memoria total: {round(props.total_memory / 1e9, 2)} GB")
        print(f"Capacidad de cómputo: ({props.major}, {props.minor})")
        print(f"Multiprocesadores: {props.multi_processor_count}")
        print(f"Max Threads por multiprocesador: {props.max_threads_per_multi_processor}")
        print(f"Warp size: {props.warp_size}")
        print(f"Compatibilidad con Tensor Cores: {'Sí' if props.major >= 7 else 'No'}")
else:
    print("No se detectó ninguna GPU CUDA.")
