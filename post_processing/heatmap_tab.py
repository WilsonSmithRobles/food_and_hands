from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

class HeatmapTab(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Left side
        heatmap_layout = QVBoxLayout()

        self.heatmap_label = QLabel()
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        heatmap_layout.addWidget(self.heatmap_label) 

        self.heatmap_progress_bar = QProgressBar()
        self.heatmap_progress_bar.setAlignment(Qt.AlignCenter)
        heatmap_layout.addWidget(self.heatmap_progress_bar)

        left_button_layout = QHBoxLayout()
        left_button1 = QPushButton("Button 1")
        left_button2 = QPushButton("Button 2")
        left_button_layout.addWidget(left_button1)
        left_button_layout.addWidget(left_button2)
        heatmap_layout.addLayout(left_button_layout)

        # Initially, hide both sides
        self.setLayout(heatmap_layout)
        self.hide()

    def load_processing_images(self, path_to_image : str):
        self.heatmap_label.setPixmap(QPixmap(path_to_image))
