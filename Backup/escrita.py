import time
import utils as fc
import numpy as np

# Definindo as variáveis necessárias
n = 1000
d = 4
m = 3
rho_sep = np.eye(4) / 4
medicoes, eta = fc.measurements(1, PLOT=False)
detp = fc.strategies_LHS(m, 2)

# Função para benchmark da versão original (sem otimização)
def original_version(n, fc):
    f1 = open("results_" + str(n) + ".txt", "w")
    f2 = open("states_" + str(n) + ".txt", "w")
    f3 = open("info_" + str(n) + ".txt", "w")

    for i in range(n):
        rho = fc.rho_mixed(d)
        f2.write(str(rho) + "\n")
        w, v, ppt = fc.Ent_cert(rho)

        input_message = [str('-'*84), "\nTarget state rho:\n", str(rho), "\nIs the state separable?\n"]
        f1.writelines(input_message)

        s = "Yes: separable state.\n" if ppt == 0 else "No: entangled state.\n"
        f1.write(s)

        f3.write(f"{i} {ppt}\n")

        P, solution, q = fc.SDP_LHS(m, 2, rho, rho_sep, eta, detp, medicoes)
        print(P, solution)

        rho_q = rho * q + (1 - q) * rho_sep
        output_message = f"Optimal value from SDP\n{q}\nResulting state\n{rho_q}\nIs the state separable?\n"
        f1.writelines(output_message)

        w, v, ppt = fc.Ent_cert(rho_q)
        s = "Yes: separable state.\n" if ppt == 0 else "No: entangled state.\n"
        f1.write(s)

        f3.write(f" {q} {ppt}\n")

    # Fechar os arquivos
    f1.close()
    f2.close()
    f3.close()

# Função para benchmark da versão otimizada (com atualização)
def updated_version(n, fc):
    with open("results_" + str(n) + ".txt", "w") as f1, \
         open("states_" + str(n) + ".txt", "w") as f2, \
         open("info_" + str(n) + ".txt", "w") as f3:

        for i in range(n):
            rho = fc.rho_mixed(d)
            f2.write(str(rho) + "\n")
            w, v, ppt = fc.Ent_cert(rho)

            input_message = [str('-'*84), "\nTarget state rho:\n", str(rho), "\nIs the state separable?\n"]
            f1.writelines(input_message)

            s = "Yes: separable state.\n" if ppt == 0 else "No: entangled state.\n"
            f1.write(s)

            f3.write(f"{i} {ppt}\n")

            P, solution, q = fc.SDP_LHS(m, 2, rho, rho_sep, eta, detp, medicoes)
            print(P, solution)

            rho_q = rho * q + (1 - q) * rho_sep
            output_message = f"Optimal value from SDP\n{q}\nResulting state\n{rho_q}\nIs the state separable?\n"
            f1.writelines(output_message)

            w, v, ppt = fc.Ent_cert(rho_q)
            s = "Yes: separable state.\n" if ppt == 0 else "No: entangled state.\n"
            f1.write(s)

            f3.write(f" {q} {ppt}\n")

# Benchmark da versão original
start_time = time.time()
original_version(n, fc)
original_duration = time.time() - start_time
print(f"Original version took {original_duration:.4f} seconds.")

# Benchmark da versão otimizada
start_time = time.time()
updated_version(n, fc)
updated_duration = time.time() - start_time
print(f"Updated version took {updated_duration:.4f} seconds.")

# Comparar as durações
performance_improvement = (original_duration - updated_duration) / original_duration * 100
print(f"Performance improvement: {performance_improvement:.2f}%")
