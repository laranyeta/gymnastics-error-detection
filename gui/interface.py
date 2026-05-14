from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton
from PyQt6.QtCore import Qt
from backend.scoring.rules import BASE_ESCORE

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UAB Escola d'Enginyeria TFG - Gymnastics Error Detection")
        self.resize(900, 600)
        
        self.e_score = BASE_ESCORE #10.0
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        #left column -> gymnast video with keypoints
        col_L = QVBoxLayout()
        self.video= QLabel("video placeholder")
        self.video.setStyleSheet("background-color: black; color: white;")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal) #video slider to see frame to frame

        col_L.addWidget(self.video, stretch=4)
        col_L.addWidget(self.frame_slider)
        
        #right column -> logs + accept/decline buttons + final e-score
        col_R = QVBoxLayout()
        self.log = QLabel("sample log")
        button_layout = QHBoxLayout()
        self.button_accept = QPushButton("accept")
        self.button_reject = QPushButton("decline")

        button_layout.addWidget(self.button_accept)
        button_layout.addWidget(self.button_reject)
        col_R.addWidget(self.log)
        col_R.addLayout(button_layout)
        col_R.addStretch() 
        
        self.score = QLabel(f"E-SCORE: {self.e_score:.1f}") 
        self.score.setAlignment(Qt.AlignmentFlag.AlignRight)
        col_R.addWidget(self.score)

        main_layout.addLayout(col_L, stretch=2)
        main_layout.addLayout(col_R, stretch=1)