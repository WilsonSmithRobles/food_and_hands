from PySide6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QLabel

class ErrorDialog(QDialog):
    def __init__(self, error_text):
        super().__init__()
        self.setWindowTitle("Error")
        QBtn = QDialogButtonBox.Ok
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.layout = QVBoxLayout()
        message = QLabel(error_text)
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
        self.exec()
