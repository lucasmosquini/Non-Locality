using BellPolytopes
using FrankWolfe
using LinearAlgebra

# Configuração do cenário
N = 3  # Tripartite (três partes)
m = 16  # Número de medições (ajustar conforme o caso)

# Vetores de medições: Posição dos vetores em um polígono regular no plano XY
println("Gerando vetores de medições para um polígono regular no plano XY com m = $m...")
measurements_vec = polygonXY_vec(m)

# Estado compartilhado: Estado GHZ tripartite
println("Definindo estado GHZ para N = $N partes...")
rho = rho_GHZ(N)


# Cálculo dos limites
println("Calculando limites superiores e inferiores para o estado GHZ...")
lower_bound_infinite, lower_bound, upper_bound, local_model, bell_inequality =
    nonlocality_threshold(measurements_vec, N; rho=rho, marg=false)

# Exibindo resultados
println("\n=== Resultados ===")

# Tensor de correlação
println("\nTensor de correlação (primeiro slice):")
p = correlation_tensor(measurements_vec, N; rho=rho, marg=false)
display(p[:, :, 1])  # Mostrando parte do tensor

# Limite inferior (com medições projetivas infinitas)
println("\nLimite inferior (medições projetivas infinitas no plano XY):")
println(lower_bound_infinite)

# Limite inferior
println("\nLimite inferior:")
println(lower_bound)
println("Modelo local gerado:")
#display(local_model.x[:, :, 1])  # Mostrando parte do modelo local gerado
println("Validação do modelo local:")
println(local_model.x == sum(local_model.weights[i] * local_model.atoms[i] for i in 1:length(local_model)))  # Deve ser true

# Limite superior
println("\nLimite superior:")
println(upper_bound)
println("Bell inequality (primeiro slice):")
#display(bell_inequality[:, :, 1])

println("\nCálculo concluído.")
