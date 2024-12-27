import numpy as np
import timeit
from numba import njit, cuda
from Ent_cert_cython import Ent_cert_cython  # Importa a função compilada

def generate_test_matrix(dim):
    """Gera uma matriz de densidade aleatória para teste."""
    A = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)  # Matriz complexa
    rho = A @ A.T.conj()  # Produto com a transposta conjugada
    rho /= np.trace(rho)  # Normaliza para ter traço 1
    return rho

# Função original para benchmark
def Ent_cert_original(rho):
    n = rho.shape
    rho_TA = np.copy(rho)
    
    for i in range(n[0] // 2):
        for j in range(n[1] // 2):
            rho_TA[i, j], rho_TA[i + n[0] // 2, j + n[1] // 2] = rho_TA[i + n[0] // 2, j + n[1] // 2], rho_TA[i, j]
    
    w, v = np.linalg.eig(rho_TA)
    ppt = 0 if np.all(w >= 0) else 1
    return w, v, ppt

# Função otimizada com Numba para CPU
@njit
def Ent_cert_optimized(rho):
    n = rho.shape[0]
    rho_TA = np.copy(rho)
    
    for i in range(n // 2):
        for j in range(n // 2):
            temp = rho_TA[i, j]
            rho_TA[i, j] = rho_TA[i + n // 2, j + n // 2]
            rho_TA[i + n // 2, j + n // 2] = temp

    w = np.linalg.eigvals(rho_TA)
    ppt = 0 if np.all(w.real >= 0) else 1
    return w, ppt

# Função paralelizada para GPU
@cuda.jit
def partial_transpose_gpu(rho, rho_TA, n):
    i, j = cuda.grid(2)
    if i < n // 2 and j < n // 2:
        temp = rho[i, j]
        rho_TA[i, j] = rho[i + n // 2, j + n // 2]
        rho_TA[i + n // 2, j + n // 2] = temp

def Ent_cert_cuda(rho):
    n = rho.shape[0]
    rho_TA = np.copy(rho)

    # Aloca a matriz na GPU
    rho_device = cuda.to_device(rho)
    rho_TA_device = cuda.to_device(rho_TA)

    # Ajuste de blocos e threads para aproveitar melhor a GPU
    threads_per_block = (32, 32)  # Definido para 32x32, pois é mais eficiente
    blocks_per_grid_x = (n // 2 + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (n // 2 + threads_per_block[1] - 1) // threads_per_block[1]

    # Forçando pelo menos 1 bloco por dimensão
    blocks_per_grid_x = max(1, blocks_per_grid_x)
    blocks_per_grid_y = max(1, blocks_per_grid_y)

    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Executa a transposição parcial na GPU
    partial_transpose_gpu[blocks_per_grid, threads_per_block](rho_device, rho_TA_device, n)

    # Copia o resultado de volta para a CPU
    rho_TA = rho_TA_device.copy_to_host()

    # Calcula autovalores e verifica separabilidade
    w = np.linalg.eigvals(rho_TA)
    ppt = 0 if np.all(w.real >= 0) else 1
    return w, ppt

# Função de benchmark
def benchmark_ent_cert(dim=100, iterations=1000):
    """Executa benchmark nas funções Ent_cert_original, Ent_cert_optimized e Ent_cert_cython."""
    
    # Gera a matriz de teste
    rho_test = generate_test_matrix(dim)
    
    # Executa o benchmark da função original
    original_time = timeit.timeit(lambda: Ent_cert_original(rho_test), number=iterations)
    s
    # Executa o benchmark da função otimizada
    optimized_time = timeit.timeit(lambda: Ent_cert_optimized(rho_test), number=iterations)
    
    # Executa o benchmark da função otimizada com Cython
    cython_time = timeit.timeit(lambda: Ent_cert_cython(rho_test), number=iterations)

    #cuda_time = timeit.timeit(lambda: Ent_cert_cuda(rho_test), number=iterations)
    
    # Resultados
    print(f"Benchmark para matriz de dimensão {dim} com {iterations} iterações:")
    print(f"Tempo da versão original: {original_time:.6f} segundos")
    print(f"Tempo da versão otimizada com Numba: {optimized_time:.6f} segundos")
    print(f"Tempo da versão otimizada com Cython: {cython_time:.6f} segundos")
    print(f"Aceleração (normal vs CPU otimizada): {original_time / optimized_time:.2f}x")
    print(f"Aceleração (normal vs Cython): {original_time / cython_time:.2f}x")
    #print(f"Aceleração (normal vs CUDA): {original_time / cuda_time:.2f}x")


# Executa o benchmark
benchmark_ent_cert(dim=4, iterations=1000000)