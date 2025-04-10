import numpy as np
from numba import cuda

@cuda.jit
def vec_add(A, B, C):
    idx = cuda.threadIdx.x
    if idx < C.size:
        C[idx] = A[idx] + B[idx]

N = 1024
size = N * np.dtype(np.float32).itemsize

# Crear vectores en CPU
A = np.random.rand(N).astype(np.float32)
B = np.random.rand(N).astype(np.float32)
C = np.zeros(N, dtype=np.float32)

# Copiar vectores a la GPU
d_A = cuda.to_device(A)
d_B = cuda.to_device(B)
d_C = cuda.device_array_like(C)

vec_add[1, N](d_A, d_B, d_C)

# Traer resultado a la CPU
C = d_C.copy_to_host()

# Mostrar resultado
print("Resultado de la suma:")
print(C)
