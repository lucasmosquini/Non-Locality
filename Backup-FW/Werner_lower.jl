using BellPolytopes
using FrankWolfe
using LinearAlgebra

# Configuração do cenário
N = 2  # Número de partes (bipartido)

# Caminho do arquivo .obj gerado
polyhedron_file = "C:/Users/lucas/OneDrive/Desktop/Mestrado/Utils/Códigos/Frank-wolfe/polyhedronisme/obj/m-43.obj"

# Carregar os vetores de medições
measurements_vec = polyhedronisme(polyhedron_file, 43)  # Ajustar o número 'm' se necessário

# Definição do estado compartilhado
rho = rho_singlet()  # Estado de Bell |ψ−⟩

# Cálculo da matriz de correlação
p = correlation_tensor(measurements_vec, N; rho=rho, marg=false)

# Fator de encolhimento (η)
η = shrinking_squared(measurements_vec; verbose=true)

# Visibilidade inicial (v0)
v0 = 0.692  # Ajuste conforme necessário

@time begin
    # Executar o algoritmo de Frank-Wolfe para encontrar o modelo local
    x, ds, primal, dual_gap, as, M, β = bell_frank_wolfe(
        p; 
        v0=v0,  # Visibilidade inicial
        verbose=3, 
        epsilon=1e-12,  # Precisão
        mode_last=-1,  # Desativa o último LMO (não necessário para lower bound)
        nb_last=10^12,  # Número máximo de iterações
        callback_interval=10^4
    )

    # Fator de correção (v2)
    v2 = 1 / (1 + norm(x - v0 * p))

    # Limite inferior (v_low = η^2 * v2 * v0)
    v_low = η^2 * v2 * v0

    println("\nResumo dos resultados:")
    println("Valor primal (objetivo): $primal")
    println("Dual gap (precisão): $dual_gap")
    println("Fator de encolhimento (η): $η")
    println("Fator de correção (v2): $v2")
    println("Limite inferior (v_c >=): $v_low")
    println("Número de estratégias usadas: ", length(ds))
end