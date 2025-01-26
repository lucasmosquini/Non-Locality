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

@time begin
    x, ds, primal, dual_gap, as, M, β = bell_frank_wolfe(
        p; 
        v0=1 / sqrt(2),  # Visibilidade inicial
        verbose=3, 
        epsilon=1e-6,  # Precisão
        mode_last=0, 
        nb_last=10^8,  # Número máximo de iterações
        callback_interval=10^5
    )
    println("\nResumo dos resultados:")
    println("Valor primal (objetivo): $primal")
    println("Dual gap (precisão): $dual_gap")
    println("Limite superior (v_c <=): ", 1 / sqrt(2))  # Modifique se necessário
    println("Número de estratégias usadas: ", length(ds))
end
