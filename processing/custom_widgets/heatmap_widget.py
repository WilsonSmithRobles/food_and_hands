import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class HeatmapWidget(QWidget):
    def __init__(self, parent=None):
        super(HeatmapWidget, self).__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.canvas)

    def plot(self, data):
        ax = self.figure.add_subplot(111)
        ax.imshow(data, cmap='hot', interpolation='nearest')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Heatmap of 1s Count in Superpixels')
        self.canvas.draw()