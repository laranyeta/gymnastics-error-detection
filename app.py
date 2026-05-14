import sys
from PyQt6.QtWidgets import QApplication
from gui.interface import MainApp

def run_app():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_app()