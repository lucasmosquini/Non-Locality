import numpy as np
import timeit
import time
from numba import njit, cuda, jit
from scipy.stats import unitary_group
from numpy import linalg as LA
from Ent_cert_cython import Ent_cert_cython  

def G_matrix(n, m):
    ''' 
    Generation of the random matrix from the Ginibre ensemble
    A complex matrix with elements having real and complex part 
    distributed with the normal distribution 
    
    input: dimensions of the Matrix G of size n x m (integers)
    output: array of matrix G of size n x m
    '''
    G = (np.random.randn(n, m) + 1j * np.random.randn(n, m)) / np.sqrt(2)
    return G

def rho_mixed(n):
    '''
    Generation of a random mixed density matrix (Bures metric)
    Input: n = dimension of the density matrix (integer)
    Output: array of density matrix 
    '''
    # Create random unitary matrix
    U = unitary_group.rvs(n)
    # Create random Ginibre matrix
    G = G_matrix(n, n)
    # Create identity matrix
    I = np.eye(n)
    # Construct density matrix
    rho = (I + U) @ G @ (G.conjugate().T) @ (I + U.conjugate().T)
    # Normalize density matrix
    rho = rho / rho.trace()
    return rho

def rho_mixed_HS(n):
    '''
    Generation of a random mixed density matrix (Hilbert-Schmidt metric)
    Input: n = dimension of the density matrix (integer)
    Output: array of density matrix 
    '''
    # Create random Ginibre matrix
    G = G_matrix(n, n)
    # Construct density matrix
    rho = G @ (G.conjugate().T)
    # Normalize density matrix
    rho = rho / rho.trace()
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
@njit(parallel=False)
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

    rho_device = cuda.to_device(rho)
    rho_TA_device = cuda.to_device(rho_TA)

    threads_per_block = (32, 32)
    blocks_per_grid_x = (n // 2 + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (n // 2 + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (max(1, blocks_per_grid_x), max(1, blocks_per_grid_y))

    partial_transpose_gpu[blocks_per_grid, threads_per_block](rho_device, rho_TA_device, n)

    rho_TA = rho_TA_device.copy_to_host()
    w = np.linalg.eigvals(rho_TA)
    ppt = 0 if np.all(w.real >= 0) else 1
    return w, ppt

# Função de benchmark
def benchmark_ent_cert(dim=4, iterations=1000, metric='Bures'):
    ent_sep = np.array([0, 0])  # Contadores para estados entrelaçados e separáveis
    st = time.time()  # Tempo inicial
    
    for i in range(iterations):
        # Gera a matriz de densidade de acordo com a métrica escolhida
        if metric == 'Bures':
            rho_test = rho_mixed(dim)
        elif metric == 'Hilbert-Schmidt':
            rho_test = rho_mixed_HS(dim)
        else:
            raise ValueError("Métrica desconhecida: Escolha entre 'Bures' ou 'Hilbert-Schmidt'")
        
        _, _, ppt = Ent_cert_original(rho_test)  # Usa a função original para o PPT
        if ppt == 1:
            ent_sep[0] += 1  # Estado entrelaçado
        else:
            ent_sep[1] += 1  # Estado separável

    et = time.time()  # Tempo final
    
    # Benchmark das funções
    rho_test = rho_mixed(dim) if metric == 'Bures' else rho_mixed_HS(dim)
    original_time = timeit.timeit(lambda: Ent_cert_original(rho_test), number=iterations)
    optimized_time = timeit.timeit(lambda: Ent_cert_optimized(rho_test), number=iterations)
    cython_time = timeit.timeit(lambda: Ent_cert_cython(rho_test), number=iterations)
    #cuda_time = timeit.timeit(lambda: Ent_cert_cuda(rho_test), number=iterations)

    print(f"Benchmark para matriz de dimensão {dim} com {iterations} iterações e métrica {metric}:")
    print(f"Tempo da versão original: {original_time:.6f} segundos")
    print(f"Tempo da versão otimizada com Numba: {optimized_time:.6f} segundos")
    print(f"Tempo da versão otimizada com Cython: {cython_time:.6f} segundos")
    print(f"Aceleração (normal vs numba): {original_time / optimized_time:.2f}x")
    print(f"Aceleração (normal vs Cython): {original_time / cython_time:.2f}x")
    #print(f"Aceleração (normal vs GPU): {original_time / cuda_time:.2f}x")
     
    # Exibe a quantidade de estados entrelaçados e separáveis e o tempo de execução
    print(f"Entanglement certification results (entangled, separable): {ent_sep}")
    print(f"Total execution time: {et - st:.2f} seconds")

# Exemplo de execução:
#benchmark_ent_cert(dim=4, iterations=1000, metric='Bures')
benchmark_ent_cert(dim=4, iterations=10000, metric='Hilbert-Schmidt')
