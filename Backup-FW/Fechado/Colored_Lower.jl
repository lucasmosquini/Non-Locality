using LinearAlgebra
using BellPolytopes
using FrankWolfe

# Função para gerar m pontos uniformemente distribuídos na esfera (Fibonacci sphere)
function fibonacci_sphere(m::Integer)
    points = zeros(Float64, 3, m)
    φ = π * (3 - sqrt(5))  # ângulo de ouro
    for i in 1:m
        y = 1 - (i - 0.5) * 2 / m
        r = sqrt(1 - y^2)
        theta = φ * (i - 1)
        x = cos(theta) * r
        z = sin(theta) * r
        points[:, i] = [x, y, z]
    end
    return points
end

# ------------------------------
# Configuração do cenário
# ------------------------------

# Sistema bipartido: N = 2 (dois qubits)
N = 2

# Número de medições
m = 80
println("Gerando um novo conjunto de medições usando Fibonacci sphere com m = $m...")
measurements_vec = fibonacci_sphere(m)  # Agora uma matriz 3 x m com pontos na esfera

# ------------------------------
# Escolha do Tipo de Estado
# ------------------------------

# Opções: "Werner", "Colored", "Isotropic"
state_type = "Werner"  # Altere para testar diferentes estados
v = 0.6875  # visibilidade

if state_type == "Werner"
    # Estado Werner: mistura do estado singlete com ruído branco I/4.
    rho_sing = rho_singlet()  # Estado singlete (|ψ⁻⟩⟨ψ⁻|)
    I4 = Matrix{Float64}(I, 4, 4)  # Identidade 4x4
    rho_noise = I4 / 4
    rho = v * rho_sing + (1 - v) * rho_noise
    println("Usando o estado Werner com visibilidade v = ", v)
    
elseif state_type == "Colored"
    # Estado com ruído colored: mistura do estado singlete com |00⟩⟨00|
    psi_minus = [0, 1, -1, 0] / sqrt(2)
    rho_psi_minus = psi_minus * psi_minus'
    rho_00 = [1.0 0.0 0.0 0.0;
              0.0 0.0 0.0 0.0;
              0.0 0.0 0.0 0.0;
              0.0 0.0 0.0 0.0]
    rho = v * rho_psi_minus + (1 - v) * rho_00
    println("Usando o estado Colored com visibilidade v = ", v)
    
elseif state_type == "Isotropic"
    # Estado Isotrópico: mistura do estado maximamente emaranhado |ϕ⁺⟩ com ruído branco I/4.
    psi_plus = [1, 0, 0, 1] / sqrt(2)
    rho_psi_plus = psi_plus * psi_plus'
    I4 = Matrix{Float64}(I, 4, 4)
    rho_noise = I4 / 4
    rho = v * rho_psi_plus + (1 - v) * rho_noise
    println("Usando o estado Isotropic com visibilidade v = ", v)
    
else
    error("Tipo de estado inválido. Escolha entre \"Werner\", \"Colored\" ou \"Isotropic\".")
end

# ------------------------------
# Cálculo da Matriz de Correlação
# ------------------------------

println("Calculando o tensor de correlação...")
p = correlation_tensor(measurements_vec, N; rho=rho, marg=false)

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
        v0 = v,
        verbose = 3,
        epsilon = 1e-12,
        mode_last = -1,
        nb_last = 10^12,
        callback_interval = 10^4
    )
    
    v2 = 1 / (1 + norm(x - v * p))
    v_low = η^2 * v2 * v
    
    println("\nResumo dos resultados:")
    println("Valor primal (objetivo): ", primal)
    println("Dual gap: ", dual_gap)
    println("Fator de encolhimento (η²): ", η)
    println("Fator de correção (v2): ", v2)
    println("Limite inferior (v_c >=): ", v_low)
    println("Número de estratégias usadas: ", length(ds))
end
