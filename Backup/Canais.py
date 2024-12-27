import numpy as np

r'''
Aqui são as funções que eu poderia implementar no código principal para 
generalizar os canais. Assim eu poderia simplesmente passar um parâmetro para outras
funções, para elas usarem esses canais
r'''

#Forma como eu achei que era descrito canais gerais
def apply_channel(r, T, t):
    """
    Aplica um canal quântico à esfera de Bloch.
    Args:
        r (ndarray): Vetor de Bloch inicial (dim 3).
        T (ndarray): Matriz de transformação (3x3).
        t (ndarray): Vetor de translação (dim 3).
    Returns:
        ndarray: Novo vetor de Bloch após a transformação.
    """
    return T @ r + t

def transform_density_matrix(r, T, t):
    """
    Aplica o canal quântico em uma matriz densidade.
    Args:
        r (ndarray): Vetor de Bloch inicial (dim 3).
        T (ndarray): Matriz de transformação (3x3).
        t (ndarray): Vetor de translação (dim 3).
    Returns:
        ndarray: Nova matriz densidade após o canal quântico.
    """
    r_new = apply_channel(r, T, t)
    rho_new = 0.5 * (np.eye(2) + r_new[0] * np.array([[0, 1], [1, 0]]) +
                     r_new[1] * np.array([[0, -1j], [1j, 0]]) +
                     r_new[2] * np.array([[1, 0], [0, -1]]))
    return rho_new


r'''
Exemplo de canais. Garantir que é CPTP
r'''

def generate_random_cptp_channel():
    """
    Gera um canal quântico aleatório que é CPTP.
    Returns:
        T (ndarray): Matriz de transformação 3x3.
        t (ndarray): Vetor de translação 3x1.
    """
    # Operadores de Kraus aleatórios
    K1 = np.random.rand(2, 2) + 1j * np.random.rand(2, 2)
    K2 = np.random.rand(2, 2) + 1j * np.random.rand(2, 2)
    
    # Normaliza para satisfazer CPTP
    K1 /= np.linalg.norm(K1)
    K2 /= np.linalg.norm(K2)

    # Extrair T e t de K1 e K2 (simplificação)
    T = np.random.rand(3, 3)  # Transformação aleatória
    T /= np.linalg.norm(T, axis=0)  # Normaliza as colunas para consistência
    t = np.random.rand(3) * 0.1  # Pequena translação
    return T, t


def thermal_channel(rho, gamma):
    """
    Aplica o canal térmico em uma matriz densidade.
    Args:
        rho (ndarray): Matriz densidade original (2x2 para qubits).
        gamma (float): Taxa de dissipação (0 <= gamma <= 1).
    Returns:
        ndarray: Nova matriz densidade após o canal.
    """
    # Operadores de Kraus
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
    
    # Aplicar o canal
    rho_new = K0 @ rho @ K0.T + K1 @ rho @ K1.T
    return rho_new

def amplitude_damping_channel(rho, p):
    """
    Aplica o canal de amplitude de fase em uma matriz densidade.
    Args:
        rho (ndarray): Matriz densidade original (2x2 para qubits).
        p (float): Probabilidade de transição (0 <= p <= 1).
    Returns:
        ndarray: Nova matriz densidade após o canal.
    """
    # Operadores de Kraus
    K0 = np.array([[1, 0], [0, np.sqrt(1 - p)]])
    K1 = np.array([[0, np.sqrt(p)], [0, 0]])
    
    # Aplicar o canal
    rho_new = K0 @ rho @ K0.T + K1 @ rho @ K1.T
    return rho_new



#Integrando ao código principal
class QuantumChannel:
    def __init__(self, T, t):
        self.T = T
        self.t = t

    def apply(self, r):
        return apply_channel(r, self.T, self.t)

    def transform_density_matrix(self, r):
        return transform_density_matrix(r, self.T, self.t)
    
r'''
Exemplo de como poderia modificar as funções atuais
r'''

def Ent_cert(rho, channel=None):
    """
    Certifica entrelaçamento usando PPT e, opcionalmente, aplica um canal quântico antes.
    Args:
        rho (ndarray): Matriz densidade.
        channel (QuantumChannel, optional): Canal quântico a ser aplicado.
    Returns:
        tuple: Autovalores, autovetores, e indicador de entrelaçamento (0 = separável, 1 = emaranhado).
    """
    if channel:
        rho = channel.transform_density_matrix(rho)

    # Resto da função permanece igual
    n = rho.shape
    rho_TA = np.zeros((n[0], n[1]), dtype=np.complex_)
    a = int(n[0] / 2)
    b = int(n[1] / 2)

    rho_TA[:a, :b] = rho[:a, :b]
    rho_TA[a:, b:] = rho[a:, b:]
    rho_TA[a:, :b] = rho[a:, :b].T
    rho_TA[:a, b:] = rho[:a, b:].T

    w, v = np.linalg.eig(rho_TA)
    ppt = 0 if all(i >= 0 for i in w) else 1
    return w, v, ppt

