using Polyhedra
using LinearAlgebra

# Função para projetar vértices na esfera unitária
function project_to_sphere(vertices)
    return [v / norm(v) for v in vertices]
end

# Função para construir o poliedro seguindo o recipe SASuSkD
function generate_polyhedron()
    println("Gerando poliedro para m = 61 com recipe SASuSkD...")

    # Passo 1: Base Icosaedro (I)
    polyhedron = Polyhedra.icosahedron(Float64)  # Corrigido: sem usar polyhedron diretamente

    # Passo 2: Aplicar operações do recipe
    # S: Projeta na esfera
    vertices = Polyhedra.points(polyhedron)
    vertices = project_to_sphere(vertices)
    polyhedron = Polyhedra.polyhedron(vrep(vertices))  # Atualizado para usar Polyhedra.polyhedron

    # A: Ambo (truncamento das arestas, cria novas faces)
    polyhedron = ambo(polyhedron)

    # S: Projeta novamente na esfera
    vertices = Polyhedra.points(polyhedron)
    vertices = project_to_sphere(vertices)
    polyhedron = Polyhedra.polyhedron(vrep(vertices))  # Atualizado para usar Polyhedra.polyhedron

    # Su: Stellate/Unstellate (operação de stellation)
    polyhedron = stellate(polyhedron)

    # Sk: Kis nas faces triangulares
    polyhedron = kis(polyhedron)

    # D: Dual (troca vértices por faces)
    polyhedron = dual(polyhedron)

    println("Poliedro gerado com sucesso!")
    return polyhedron
end

# Função para exportar o poliedro como arquivo .obj
function export_polyhedron(polyhedron, file_path)
    println("Exportando poliedro para o arquivo: $file_path")
    open(file_path, "w") do io
        write_obj(io, polyhedron)
    end
    println("Exportação concluída!")
end

# Caminho para salvar o poliedro
output_path = "C:/Users/lucas/OneDrive/Desktop/Mestrado/Utils/Códigos/Frank-wolfe/polyhedronisme/obj/polyhedron_m61.obj"

# Executar geração e exportação
polyhedron = generate_polyhedron()
export_polyhedron(polyhedron, output_path)
