import numpy as np
cimport numpy as np

def Ent_cert_cython(rho):
    n = rho.shape[0]
    rho_TA = rho.copy()

    for i in range(n // 2):
        for j in range(n // 2):
            temp = rho_TA[i, j]
            rho_TA[i, j] = rho_TA[i + n // 2, j + n // 2]
            rho_TA[i + n // 2, j + n // 2] = temp

    w = np.linalg.eigvals(rho_TA)
    ppt = 0 if np.all(w.real >= 0) else 1
    return w, ppt
