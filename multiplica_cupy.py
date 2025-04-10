import cupy as cp

N = 1024

A = cp.random.rand(N).astype(cp.float32)
B = cp.random.rand(N).astype(cp.float32)
C = A + B  # ¡Ni kernel necesitas!

# Si querés traerlo a la CPU:
C_host = cp.asnumpy(C)

print("Resultado con CuPy:")
print(C_host)
