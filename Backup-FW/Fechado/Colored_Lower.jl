using BellPolytopes
using FrankWolfe
using LinearAlgebra

# ------------------------------
# Configuração do cenário
# ------------------------------

# Sistema bipartido: N = 2 (dois qubits)
N = 2

# Caminho do arquivo do poliedro 
polyhedron_file = "C:/Users/lucas/OneDrive/Desktop/Mestrado/Utils/Códigos/Frank-wolfe/polyhedronisme/obj/m-43.obj"

# Carregar os vetores de medições do poliedro
println("Carregando vetores de medições a partir do arquivo do poliedro...")
measurements_vec = polyhedronisme(polyhedron_file, 43)

# ------------------------------
# Escolha do Tipo de Estado
# ------------------------------

# Opções: "Werner", "Colored", "Isotropic"
state_type = "Isotropic"   # Altere conforme necessário
v = 0.5  # visibilidade

if state_type == "Werner"
    # Estado Werner: mistura do estado singlete com ruído branco I/4.
    rho_sing = rho_singlet()  # Estado singlete (|ψ⁻⟩⟨ψ⁻|)
    I4 = Matrix{Float64}(I, 4, 4)  # Identidade 4x4
    rho_noise = I4 / 4
    rho = v * rho_sing + (1 - v) * rho_noise
    println("Usando o estado Werner com visibilidade v = ", v)

elseif state_type == "Colored"
    # Estado com ruído colorido: mistura do estado singlete com |00⟩⟨00|
    psi_minus = [0, 1, -1, 0] / sqrt(2)  # Estado singlete |ψ⁻⟩ em representação vetorial (coluna)
    rho_psi_minus = psi_minus * psi_minus'  # |ψ⁻⟩⟨ψ⁻|
    # Estado |00⟩⟨00|
    rho_00 = [1.0 0.0 0.0 0.0;
              0.0 0.0 0.0 0.0;
              0.0 0.0 0.0 0.0;
              0.0 0.0 0.0 0.0]
    rho = v * rho_psi_minus + (1 - v) * rho_00
    println("Usando o estado com ruído colored com visibilidade v = ", v)

elseif state_type == "Isotropic"
    # Estado Isotrópico: mistura de um estado maximamente emaranhado com ruído branco.
    # Para dois qubits, usamos o estado de Bell |ϕ⁺⟩ = (|00⟩ + |11⟩)/√2.
    psi_plus = [1, 0, 0, 1] / sqrt(2)
    rho_psi_plus = psi_plus * psi_plus'
    I4 = Matrix{Float64}(I, 4, 4)
    rho_noise = I4 / 4
    rho = v * rho_psi_plus + (1 - v) * rho_noise
    println("Usando o estado isotrópico com visibilidade v = ", v)

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
        v0 = v,                # Usa a visibilidade como parâmetro inicial
        verbose = 3,
        epsilon = 1e-12,
        mode_last = -1,        # Desativa o último LMO (como para upper bound)
        nb_last = 10^12,       # Número máximo de iterações
        callback_interval = 10^4
    )

    # Cálculo do fator de correção (v2)
    v2 = 1 / (1 + norm(x - v * p))
    
    # Cálculo do limite inferior para o estado escolhido
    v_low = η^2 * v2 * v
    
    println("\nResumo dos resultados:")
    println("Valor primal (objetivo): ", primal)
    println("Dual gap (precisão): ", dual_gap)
    println("Fator de encolhimento (η²): ", η)
    println("Fator de correção (v2): ", v2)
    println("Limite inferior (v_c >=): ", v_low)
    println("Número de estratégias usadas: ", length(ds))
end
