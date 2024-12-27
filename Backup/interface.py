from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import numpy as np
import importlib.util
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Carregar os módulos utils.py e OOP.py com o caminho corrigido
utils_path = r"C:/Users/lucas/OneDrive/Desktop/Mestrado/Utils/Códigos/utils.py"
spec_utils = importlib.util.spec_from_file_location("utils", utils_path)
utils = importlib.util.module_from_spec(spec_utils)
spec_utils.loader.exec_module(utils)

OOP_path =  r"C:/Users/lucas/OneDrive/Desktop/Mestrado/Utils/Códigos/OOP.py"
spec_oop = importlib.util.spec_from_file_location("OOP", OOP_path)
oop = importlib.util.module_from_spec(spec_oop)
spec_oop.loader.exec_module(oop)

# Definindo algumas matrizes de densidade de teste
test_matrices = {
    "Bell": np.array([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]]),
    "GHZ": np.array([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]]),
    "W": np.array([[1/3, 0, 0, 1/3], [0, 1/3, 0, 0], [0, 0, 1/3, 0], [1/3, 0, 0, 1/3]]),
    "Mixed": np.array([[0.25, 0, 0, 0.25], [0, 0.25, 0.25, 0], [0, 0.25, 0.25, 0], [0.25, 0, 0, 0.25]])
}

class EntanglementCertifierApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Configurações da janela
        self.setWindowTitle('Certificação de Emaranhamento Quântico')
        self.setGeometry(100, 100, 800, 600)
        
        # Layout principal
        layout = QtWidgets.QVBoxLayout()

        # Seletor de matrizes de densidade de teste
        self.matrix_selection = QtWidgets.QComboBox(self)
        self.matrix_selection.addItems(test_matrices.keys())
        self.matrix_selection.currentIndexChanged.connect(self.load_test_matrix)
        layout.addWidget(QtWidgets.QLabel("Selecione uma Matriz de Densidade de Teste:"))
        layout.addWidget(self.matrix_selection)

        # Campo de entrada para matriz de densidade
        self.density_matrix_input = QtWidgets.QTextEdit(self)
        self.density_matrix_input.setPlaceholderText("Insira a matriz de densidade aqui...")
        layout.addWidget(QtWidgets.QLabel("Matriz de Densidade (ou selecione uma matriz de teste acima):"))
        layout.addWidget(self.density_matrix_input)

        # Botão para iniciar o cálculo
        self.calculate_button = QtWidgets.QPushButton("Calcular Emaranhamento", self)
        self.calculate_button.clicked.connect(self.calculate_entanglement)
        layout.addWidget(self.calculate_button)

        # Área de exibição de resultados
        self.result_display = QtWidgets.QTextEdit(self)
        self.result_display.setReadOnly(True)
        layout.addWidget(QtWidgets.QLabel("Resultado do Cálculo:"))
        layout.addWidget(self.result_display)

        # Gráfico da Matriz de Densidade
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Definir layout principal
        self.setLayout(layout)

    def load_test_matrix(self):
        # Carrega a matriz de teste selecionada e exibe no campo de entrada
        matrix_name = self.matrix_selection.currentText()
        matrix = test_matrices[matrix_name]
        self.density_matrix_input.setText(str(matrix.tolist()))

    def calculate_entanglement(self):
        # Obtendo a matriz de densidade da entrada
        matrix_input_text = self.density_matrix_input.toPlainText()

        # Processando a entrada para criar a matriz de densidade
        try:
            density_matrix = np.array(eval(matrix_input_text))
            result = self.check_entanglement(density_matrix)  # Verifica emaranhamento
            self.result_display.setText(result)
            self.display_density_matrix(density_matrix)
        except Exception as e:
            self.result_display.setText(f"Erro no cálculo: {e}")

    def check_entanglement(self, matrix):
        # Verifica se o estado é separável ou emaranhado
        quantum_state = oop.QuantumState(matrix.shape[0])
        quantum_state.rho = matrix
        if quantum_state.is_separable():
            return "O estado é separável."
        else:
            return "O estado é emaranhado."

    def display_density_matrix(self, matrix):
        self.ax.clear()
        cax = self.ax.matshow(matrix, cmap='viridis')
        self.figure.colorbar(cax)
        self.canvas.draw()

# Executando a aplicação
app = QtWidgets.QApplication(sys.argv)
window = EntanglementCertifierApp()
window.show()
sys.exit(app.exec_())
