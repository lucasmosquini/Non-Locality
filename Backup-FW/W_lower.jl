using BellPolytopes
using FrankWolfe
using LinearAlgebra

function calculate_bounds(state_type::String, m::Int; v0::Float64)
    # Configuração do cenário
    N = 3  # Tripartite (três partes)
    println("Estado escolhido: $state_type")
    println("Número de medições (m): $m")

    # Escolha do poliedro e definição do estado compartilhado
    if state_type == "GHZ"
        println("Usando medições no plano XY...")
        measurements_vec = polygonXY_vec(m)  # Vetores no plano XY
        rho = rho_GHZ(N)  # Estado GHZ
        marg = false  # Não inclui marginais
    elseif state_type == "W"
        println("Usando poliedro pentakis dodecahedron...")
        polyhedron_file = "C:/Users/lucas/OneDrive/Desktop/Mestrado/Utils/Códigos/Frank-wolfe/polyhedronisme/obj/pentakis_dodecahedron.obj"
        measurements_vec = polyhedronisme(polyhedron_file, m)
        rho = rho_W(N)  # Estado W
        marg = true  # Inclui marginais
    else
        error("Estado desconhecido: $state_type. Escolha entre 'GHZ' ou 'W'.")
    end

    # Calcular tensor de correlação
    println("Calculando o tensor de correlação...")
    p = correlation_tensor(measurements_vec, N; rho=rho, marg=marg)

    # Rodar o algoritmo de Frank-Wolfe para limites superiores e inferiores
    println("Rodando o algoritmo para calcular limites...")
    x, ds, primal, dual_gap, as, M, β = bell_frank_wolfe(
        p; 
        v0=v0,
        epsilon=1e-6,
        marg=marg,
        verbose=3
    )

    # Fator de redução ν²
    ν² = 1 - norm(x - v0 * p)^2

    # Calcular limites
    η = cos(pi / (2 * m))  # Fator de encolhimento do poliedro
    v_lower = η^N * ν² * v0
    v_upper = primal  # O valor primal é o limite superior

    # Exibir resultados
    println("\n=== Resultados ===")
    println("Limite inferior calculado: $v_lower")
    println("Limite superior calculado: $v_upper")
    println("Fator de redução (ν²): $ν²")
    println("Fator de encolhimento (η): $η")
end

# Exemplo de execução
state_type = "GHZ"  # Ou "GHZ"
m = 10  # Número de medições (ajustar conforme o estado)
v0 = state_type == "GHZ" ? 0.5 : 0.525  # Visibilidade inicial conforme o estado
calculate_bounds(state_type, m; v0=v0)
