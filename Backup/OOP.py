import utils as fc
import numpy as np
import cProfile
import pstats


class QuantumState:
    def __init__(self, dimension):
        self.dimension = dimension
        self.rho_sep = np.eye(dimension) / dimension  # Estado separável
        self.rho = None

    def generate_mixed_state(self):
        # Gera um estado misto
        self.rho = fc.rho_mixed(self.dimension)
        return self.rho

    def is_separable(self):
        w, v, ppt = fc.Ent_cert(self.rho)
        return ppt == 0  # Se ppt == 0, é separável

    def update_state(self, q):
        # Atualiza o estado com a estratégia de interpolação
        rho_q = self.rho * q + (1 - q) * self.rho_sep
        self.rho = rho_q
        return self.rho

class MeasurementPolytope:
    def __init__(self, measurement_type):
        self.measurement_type = measurement_type
        self.medicoes, self.eta = fc.measurements(self.measurement_type, PLOT=True)
        self.m = int(self.medicoes.shape[0] / 2)

    def get_measurements(self):
        return self.medicoes, self.eta, self.m

class DeterministicStrategy:
    def __init__(self, m, dimension):
        self.strategies = fc.strategies_LHS(m, dimension)

    def get_strategies(self):
        return self.strategies

class QuantumSimulation:
    def __init__(self, n, dimension, measurement_type):
        self.n = n
        self.dimension = dimension
        self.measurement_type = measurement_type
        self.state = QuantumState(dimension)
        self.polytope = MeasurementPolytope(measurement_type)
        self.strategy = DeterministicStrategy(self.polytope.m, 2)

        # Arquivos para armazenar os resultados
        self.f1 = open(f"results_{n}.txt", "w")
        self.f2 = open(f"states_{n}.txt", "w")
        self.f3 = open(f"info_{n}.txt", "w")

    def run(self):
        for i in range(self.n):
            # Gerar um estado misto
            rho = self.state.generate_mixed_state()
            self.f2.write(str(rho) + "\n")

            # Verificar se o estado é separável
            separable = self.state.is_separable()

            input_message = [str('-' * 84), "\nTarget state rho:\n", str(rho), "\nIs the state separable?\n"]
            self.f1.writelines(input_message)
            self.f1.write("Yes: separable state.\n" if separable else "No: entangled state.\n")
            self.f3.write(f"{i} {1 if separable else 0}\n")

            # Executar o SDP e atualizar o estado
            medicoes, eta, m = self.polytope.get_measurements()
            
            use_svd = False  # Altere para False para desativar o SVD
            if use_svd:
                medicoes = fc.apply_svd_to_measurements(medicoes)
                
                
            P, solution, q = fc.SDP_LHS(m, 2, rho, self.state.rho_sep, eta, self.strategy.get_strategies(), medicoes)
            print(P)
            print(solution)

            rho_q = self.state.update_state(q)
            output_message = ("Optimal value from SDP\n", str(q), "\nResulting state\n", str(rho_q), "\nIs the state separable?\n")
            self.f1.writelines(output_message)

            separable_q = self.state.is_separable()
            self.f1.write("Yes: separable state.\n" if separable_q else "No: entangled state.\n")
            self.f3.write(f" {q} {1 if separable_q else 0}\n")

        # Fechar os arquivos
        self.f1.close()
        self.f2.close()
        self.f3.close()

# Inicialização e execução da simulação
n = 100 # Número de iterações
dimension = 4  # Dimensão do estado quântico
measurement_type = 1 # Tipo de medição

simulation = QuantumSimulation(n, dimension, measurement_type)
simulation.run()


print(f"Tempo total para SDP_LHS: {fc.SDP_total_time:.2f} segundos")



def main(density_matrix):
    # Defina os parâmetros conforme necessário (por exemplo, número de iterações, dimensão, tipo de medição)
    n = 100  # Número de iterações
    dimension = len(density_matrix)  # Dimensão da matriz de densidade (4x4)
    measurement_type = 2  # Tipo de medição

    simulation = QuantumSimulation(n, dimension, measurement_type)
    simulation.run()
    return "Simulação concluída"


if __name__ == "__main__":
    profiler = cProfile.Profile()

    # Abra os arquivos antes do profiling
    simulation = QuantumSimulation(n, dimension, measurement_type)
    simulation.f1 = open(f"results_{n}.txt", "w")
    simulation.f2 = open(f"states_{n}.txt", "w")
    simulation.f3 = open(f"info_{n}.txt", "w")

    profiler.enable()
    simulation.run()  # Executa a simulação com profiling
    profiler.disable()

    # Fecha os arquivos após o profiling
    simulation.f1.close()
    simulation.f2.close()
    simulation.f3.close()

    # Exibe as estatísticas do profiling
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(10)
    profiler.dump_stats("profile_results.prof")
    

### python oop.py
### snakeviz profile_results.prof



