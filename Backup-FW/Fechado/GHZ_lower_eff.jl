using BellPolytopes
using LinearAlgebra

# Número de partes (tripartido para GHZ-3)
N = 3  

# Caminho do arquivo .obj do poliedro (mude para 43 ou 33 conforme necessário)
polyhedron_file = "C:/Users/lucas/OneDrive/Desktop/Mestrado/Utils/Códigos/Frank-wolfe/polyhedronisme/obj/m-43.obj"

# Carregar os vetores de medições
measurements_vec = polyhedronisme(polyhedron_file, 43)

# Definir o estado GHZ
rho = rho_GHZ(N)  

# Calcular a matriz de correlação
p = correlation_tensor(measurements_vec, N; rho=rho, marg=false)

# Fator de encolhimento η²
η² = shrinking_squared(measurements_vec; verbose=true)
println("Fator de encolhimento (η²): ", η²)

# Visibilidade inicial (v0)
v0 = 0.5  # Conforme o artigo

# Executar Frank-Wolfe com poucas iterações para encontrar x aproximado
x, _, _, _, _, _, _ = bell_frank_wolfe(
    p; 
    v0=v0,  
    verbose=1,  
    epsilon=1e-6,  # Precisão menor para rodar mais rápido
    mode_last=-1,  
    nb_last=10^4  # Menos iterações para rodar rápido
)

# Calcular o fator de correção (v2)
v2 = 1 / (1 + norm(x - v0 * p))
println("Fator de correção (v2): ", v2)

# Calcular o limite inferior (v_c)
v_low = η² * v2 * v0
println("Estimativa do limite inferior (v_c ≥): ", v_low)
