from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt
import gui.style as css

class DeductionWidget(QFrame): #widget for each deduction reason in the log, with accept/reject buttons
    def __init__(self, frame_idx, reason_idx, reason_obj, main_app):
        super().__init__()
        self.main_app = main_app
        self.frame_idx = frame_idx
        self.reason_idx = reason_idx
        self.reason_obj = reason_obj

        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        
        self.lbl_text = QLabel(reason_obj["text"])
        self.lbl_text.setWordWrap(True)
        self.lbl_text.setStyleSheet(css.LBL_REASON_STYLE)
        
        btn_layout = QHBoxLayout()
        self.btn_acc = QPushButton("Accept")
        self.btn_acc.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_rej = QPushButton("Decline")
        self.btn_rej.setCursor(Qt.CursorShape.PointingHandCursor)
        
        btn_layout.addWidget(self.btn_acc)
        btn_layout.addWidget(self.btn_rej)
        
        layout.addWidget(self.lbl_text)
        layout.addLayout(btn_layout)
        
        self.btn_acc.clicked.connect(self.accept_deduction)
        self.btn_rej.clicked.connect(self.reject_deduction)
        self.update_ui()

    def update_ui(self):
        status = self.reason_obj["status"]
        if status == "accepted":
            self.setStyleSheet(css.DEDUCTION_ACCEPTED)
            self.btn_acc.setVisible(False)
            self.btn_rej.setVisible(False)
        elif status == "rejected":
            self.setStyleSheet(css.DEDUCTION_REJECTED)
            self.btn_acc.setVisible(False)
            self.btn_rej.setVisible(False)
        else:
            self.setStyleSheet(css.DEDUCTION_PENDING)
            self.btn_acc.setStyleSheet(css.BTN_ACC_STYLE)
            self.btn_rej.setStyleSheet(css.BTN_REJ_STYLE)
            self.btn_acc.setVisible(True)
            self.btn_rej.setVisible(True)

    def accept_deduction(self):
        self.main_app.logic.accept_deduction(self.frame_idx, self.reason_idx)
        self.main_app.update_gui_after_action()

    def reject_deduction(self):
        self.main_app.logic.reject_deduction(self.frame_idx, self.reason_idx)
        self.main_app.update_gui_after_action()