using BellPolytopes
using FrankWolfe
using LinearAlgebra

# Configuração do cenário
N = 2  # Número de partes (bipartido)

# Definição dos vetores de medições (ajustar para diferentes arquivos .obj com mais vértices)
polyhedron_file = "C:/Users/lucas/OneDrive/Desktop/Mestrado/Utils/Códigos/Frank-wolfe/polyhedronisme/obj/polyhedronisme-SASuSAuO.obj"
measurements_vec = polyhedronisme(polyhedron_file, 33) 

# Definição do estado compartilhado
rho = rho_singlet()  # Estado de Bell |ψ−⟩

# Cálculo da matriz de correlação
p = correlation_tensor(measurements_vec, N; rho=rho, marg=false)

# Configuração do Frank-Wolfe
max_iterations = 10^7  # Aumentar o número de iterações para melhorar a precisão
epsilon = 1e-6  # Reduzir a tolerância para maior precisão

@time begin
    x, ds, primal, dual_gap, as, M, β = bell_frank_wolfe(
        p; 
        v0=1 / sqrt(2), 
        verbose=3, 
        epsilon=epsilon, 
        mode_last=0, 
        nb_last=max_iterations, 
        callback_interval=10^5
    )
    println("\nResumo dos resultados:")
    println("Valor primal (objetivo): $primal")
    println("Dual gap (precisão): $dual_gap")
    println("Limite superior (v_c <=): ", 1 / sqrt(2)) # Modifique se necessário
    println("Número de estratégias usadas: ", length(ds))
end
