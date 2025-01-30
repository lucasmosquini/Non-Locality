using BellPolytopes
using FrankWolfe
using LinearAlgebra

# Configuração do cenário para o estado GHZ
N = 3  # Número de partes (tripartido, no caso do GHZ)
m = 43  # ou 33, dependendo do poliedro disponível

# Caminho do arquivo .obj gerado para o poliedro
polyhedron_file = "C:/Users/lucas/OneDrive/Desktop/Mestrado/Utils/Códigos/Frank-wolfe/polyhedronisme/obj/m-43.obj"

# Carregar os vetores de medições do poliedro
measurements_vec = polyhedronisme(polyhedron_file, m)

# Definição do estado GHZ (no caso, usaremos um estado GHZ típico de 3 partes)
rho = rho_GHZ(N)

# Cálculo da matriz de correlação (com marginais, como descrito no suplemento)
p = correlation_tensor(measurements_vec, N; rho=rho, marg=true)

# Fator de encolhimento (η) conforme descrito no suplemento
η = shrinking_squared(measurements_vec; verbose=true)

# Visibilidade inicial (v0), ajustada de acordo com o suplemento
v0 = 0.5  # Ajuste conforme necessário

@time begin
    # Executar o algoritmo de Frank-Wolfe para encontrar o modelo local
    x, ds, primal, dual_gap, as, M, β = bell_frank_wolfe(
        p; 
        v0=v0,  # Visibilidade inicial
        verbose=3, 
        epsilon=1e-7,  # Precisão
        mode_last=-1,  # Desativa o último LMO (não necessário para lower bound)
        nb_last=10^6,  # Número máximo de iterações
        callback_interval=10^4
    )

    # Fator de correção (v2)
    v2 = 1 / (1 + norm(x - v0 * p))

    # Limite inferior (v_low = η^N * v2 * v0)
    v_low = η^N * v2 * v0

    println("\nResumo dos resultados:")
    println("Valor primal (objetivo): $primal")
    println("Dual gap (precisão): $dual_gap")
    println("Fator de encolhimento (η): $η")
    println("Fator de correção (v2): $v2")
    println("Limite inferior (v_c >=): $v_low")
    println("Número de estratégias usadas: ", length(ds))
end
