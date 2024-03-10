import sys
from PySide6.QtWidgets import QApplication, QWidget, QMainWindow, QHBoxLayout
from PySide6.QtGui import QAction

from post_processing.heatmap_tab import HeatmapTab
from processing.processing_tab import ProcessingTab

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("TFM GUI")

        # Toolbar
        toolbar = self.addToolBar("Toolbar")

        # Actions for showing/hiding sides
        show_processing_tab = QAction("Show Processing Tab", self)
        show_processing_tab.triggered.connect(self.show_processing)
        toolbar.addAction(show_processing_tab)

        show_heatmap_tab = QAction("Show Right Side", self)
        show_heatmap_tab.triggered.connect(self.show_heatmap)
        toolbar.addAction(show_heatmap_tab)

        # Central layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.central_layout = QHBoxLayout(central_widget)

        # Building everything
        self.processing_tab = ProcessingTab()
        self.heatmap_tab = HeatmapTab()
        self.central_layout.addWidget(self.processing_tab, stretch=1)
        self.central_layout.addWidget(self.heatmap_tab, stretch=1)

        self.setGeometry(100, 100, 800, 600)  # Adjust the geometry as needed
        self.show()

    def show_processing(self):
        self.heatmap_tab.hide()
        self.processing_tab.show()

    def show_heatmap(self):
        self.processing_tab.hide()
        self.heatmap_tab.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec())
