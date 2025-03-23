using BellPolytopes
using FrankWolfe
using LinearAlgebra

# ------------------------------
# Configuração do cenário Werner
# ------------------------------

# Sistema bipartido: N = 2 (dois qubits)
N = 2

# Caminho do arquivo do poliedro 
polyhedron_file = "C:/Users/lucas/OneDrive/Desktop/Mestrado/Utils/Códigos/Frank-wolfe/polyhedronisme/obj/m-43.obj"

# Carregar os vetores de medições do poliedro
println("Carregando vetores de medições a partir do arquivo do poliedro...")
measurements_vec = polyhedronisme(polyhedron_file, 43)

# ------------------------------
# Construção do Estado Werner
# ------------------------------

# Obtém o estado singlete (estado de Bell puro |ψ⁻⟩⟨ψ⁻|)
rho_sing = rho_singlet()

# Cria o operador identidade normalizado (para dois qubits: dimensão 4x4)
I4 = Matrix{Float64}(I, 4, 4)
rho_noise = I4 / 4

# Define a visibilidade v (por exemplo, v = 0.6875)
v = 0.6875

# Constrói o estado Werner como uma combinação convexa:
# ρ_Werner = v * ρ_sing + (1 - v) * ρ_noise
rho_Werner = v * rho_sing + (1 - v) * rho_noise
println("Usando o estado Werner com visibilidade v = ", v)

# ------------------------------
# Cálculo da Matriz de Correlação
# ------------------------------

println("Calculando o tensor de correlação...")
p = correlation_tensor(measurements_vec, N; rho=rho_Werner, marg=false)

# ------------------------------
# Cálculo do Fator de Encolhimento
# ------------------------------

η = shrinking_squared(measurements_vec; verbose=true)
println("Fator de encolhimento (η²): ", η)

# ------------------------------
# Execução do Algoritmo Frank-Wolfe
# ------------------------------

@time begin
    x, ds, primal, dual_gap, as, M, β = bell_frank_wolfe(
        p;
        v0 = v,                # Usando o mesmo v como visibilidade inicial
        verbose = 3,
        epsilon = 1e-12,
        mode_last = -1,        # Desativa o último LMO para o cálculo do upper bound
        nb_last = 10^12,       # Número máximo de iterações
        callback_interval = 10^4
    )

    # Cálculo do fator de correção (v2)
    v2 = 1 / (1 + norm(x - v * p))
    
    # Cálculo do limite inferior para o estado Werner
    v_low = η^2 * v2 * v
    
    println("\nResumo dos resultados:")
    println("Valor primal (objetivo): ", primal)
    println("Dual gap (precisão): ", dual_gap)
    println("Fator de encolhimento (η²): ", η)
    println("Fator de correção (v2): ", v2)
    println("Limite inferior (v_c >=): ", v_low)
    println("Número de estratégias usadas: ", length(ds))
end
