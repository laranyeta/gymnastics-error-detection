import cv2
import gui.style as css

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QSlider, QPushButton, QFileDialog, 
                             QMessageBox, QScrollArea)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QImage, QPixmap, QAction, QIcon
from backend.rnn.score import generate_skeleton_canvas
from gui.components import DeductionWidget
from gui.logic import AppLogic

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UAB Escola d'Enginyeria TFG - Gymnastics Error Detection")
        self.resize(1200, 750)
        
        self.logic = AppLogic()
        
        self.video_cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.is_playing = False
        self.current_frame = 0  
        self.current_view_mode = "Video"
        
        self.create_menu_bar()
        self.setup_ui()

    def create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')

        load_video_action = QAction('Load Video (.mp4/.avi)', self)
        load_video_action.triggered.connect(self.load_video)
        file_menu.addAction(load_video_action)

        self.load_json_action = QAction('Load Data (.json)', self)
        self.load_json_action.triggered.connect(self.load_json)
        self.load_json_action.setEnabled(False) 
        file_menu.addAction(self.load_json_action)

    def setup_ui(self):
        central_widget = QWidget()
        central_widget.setStyleSheet(css.MAIN_BG_STYLE)
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15) #space between columns

        #left
        left_container = QWidget()
        left_container.setStyleSheet(css.LEFT_COL_STYLE)
        col_L = QVBoxLayout(left_container)

        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(5) 
        
        self.btn_mode_video = QPushButton("Original Video")
        self.btn_mode_video.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_mode_video.setCheckable(True)
        self.btn_mode_video.setChecked(True)
        self.btn_mode_video.clicked.connect(lambda: self.switch_view_mode("Video"))
        self.btn_mode_video.setStyleSheet(css.TAB_STYLE)
        
        self.btn_mode_skeleton = QPushButton("Skeleton")
        self.btn_mode_skeleton.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_mode_skeleton.setCheckable(True)
        self.btn_mode_skeleton.clicked.connect(lambda: self.switch_view_mode("Skeleton"))
        self.btn_mode_skeleton.setStyleSheet(css.TAB_STYLE)
        
        mode_layout.addWidget(self.btn_mode_video, stretch=1)
        mode_layout.addWidget(self.btn_mode_skeleton, stretch=1)
        
        self.video = QLabel("Go to File > Load Video to start")
        self.video.setStyleSheet(css.VIDEO_STYLE)
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumSize(640, 480)
        
        controls_layout = QHBoxLayout()

        def create_icon_button(icon_path, callback): #helper function button as icon
            btn = QPushButton()
            btn.setIcon(QIcon(icon_path))
            btn.setIconSize(QSize(20, 20))
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(callback)
            btn.setFixedSize(40, 40)
            btn.setStyleSheet("background-color: transparent;")
            return btn
        
        self.btn_prev_err = create_icon_button("gui/assets/prev.png", self.jump_prev_error)
        self.btn_prev_err.setEnabled(False)
        
        self.btn_play_pause = create_icon_button("gui/assets/play.png", self.toggle_playback)
        self.btn_play_pause.setEnabled(False)
        
        self.btn_next_err = create_icon_button("gui/assets/next.png", self.jump_next_error)
        self.btn_next_err.setEnabled(False)
        
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.sliderMoved.connect(self.set_frame_position)
        
        self.lbl_frame_counter = QLabel("Frame: 0/0")
        
        controls_layout.addWidget(self.btn_prev_err)
        controls_layout.addWidget(self.btn_play_pause)
        controls_layout.addWidget(self.btn_next_err)
        controls_layout.addWidget(self.frame_slider)
        controls_layout.addWidget(self.lbl_frame_counter)

        col_L.addLayout(mode_layout)
        col_L.addWidget(self.video, stretch=4)
        col_L.addLayout(controls_layout)
        
        # --- CONTENIDOR DRETA ---
        right_container = QWidget()
        right_container.setStyleSheet(css.RIGHT_COL_STYLE)
        col_R = QVBoxLayout(right_container)
        
        logs_header_layout = QHBoxLayout()
        self.lbl_logs_title = QLabel("<b>Deductions Logs</b>")
        self.lbl_logs_title.setStyleSheet(css.TITLE_STYLE)
        
        self.btn_undo = QPushButton("Undo")
        self.btn_undo.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_undo.setFixedWidth(70)
        self.btn_undo.clicked.connect(self.undo_action)
        
        self.btn_redo = QPushButton("Redo")
        self.btn_redo.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_redo.setFixedWidth(70)
        self.btn_redo.clicked.connect(self.redo_action)
        
        logs_header_layout.addWidget(self.lbl_logs_title)
        logs_header_layout.addStretch()
        logs_header_layout.addWidget(self.btn_undo)
        logs_header_layout.addWidget(self.btn_redo)
        
        self.lbl_acrobatic_info = QLabel("<b>Detected Acrobatic:</b> None")
        self.lbl_acrobatic_info.setStyleSheet("font-size: 16px;")
        self.lbl_confidence_info = QLabel("<b>Confidence:</b> None")
        self.lbl_confidence_info.setStyleSheet("font-size: 14px; color: #555;")
        
        self.log_scroll = QScrollArea()
        self.log_scroll.setWidgetResizable(True)
        self.log_scroll.setStyleSheet(css.SCROLL_AREA_STYLE + css.SCROLLBAR_STYLE)
        
        self.log_container = QWidget()
        self.log_container.setStyleSheet(css.LOG_CONTAINER_STYLE)
        self.log_layout = QVBoxLayout(self.log_container)
        self.log_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.log_scroll.setWidget(self.log_container)
        
        self.btn_reject_all = QPushButton("False Positive (Discard frame)")
        self.btn_reject_all.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reject_all.setStyleSheet(css.BTN_REJECT_ALL_STYLE)
        self.btn_reject_all.clicked.connect(self.reject_all_deductions)

        self.score_title = QLabel(f"Final E-Score")
        self.score_title.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.score_title.setStyleSheet(css.SCORETITLE_STYLE)

        self.score = QLabel(f"{self.logic.e_score:.1f}")
        self.score.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.score.setStyleSheet(css.SCORE_STYLE)

        col_R.addLayout(logs_header_layout)
        col_R.addWidget(self.lbl_acrobatic_info)
        col_R.addWidget(self.lbl_confidence_info)
        col_R.addWidget(self.log_scroll, stretch=1)
        col_R.addWidget(self.btn_reject_all)
        col_R.addWidget(self.score_title)
        col_R.addWidget(self.score)

        main_layout.addWidget(left_container, stretch=5)
        main_layout.addWidget(right_container, stretch=5)

        self.lbl_acrobatic_info.hide()
        self.lbl_confidence_info.hide()
        self.btn_reject_all.hide()
        self.btn_undo.hide()
        self.btn_redo.hide()
        
        self.update_undo_redo_buttons()

    def switch_view_mode(self, mode):
        self.current_view_mode = mode
        self.btn_mode_video.setChecked(mode == "Video")
        self.btn_mode_skeleton.setChecked(mode == "Skeleton")
        self.refresh_display()

    # --- CÀRREGA DE FITXERS ---
    def load_video(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select video", "", "videos (*.mp4 *.avi)")
        if filename:
            if self.video_cap: self.video_cap.release()
            self.video_cap = cv2.VideoCapture(filename)
            self.frame_slider.setMaximum(int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
            self.frame_slider.setEnabled(True)
            self.btn_play_pause.setEnabled(True)
            self.load_json_action.setEnabled(True) 
            self.set_frame_position(0)
            
    def load_json(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select JSON", "", "JSON (*.json)")
        if filename:
            try:
                # El cervell fa la feina
                total_errors = self.logic.load_json_data(filename)
                self.update_gui_after_action()
                
                # Despertem la UI
                self.lbl_logs_title.show()
                self.btn_undo.show()
                self.btn_redo.show()
                self.lbl_acrobatic_info.show()
                self.lbl_confidence_info.show()
                self.btn_reject_all.show()
                
                if total_errors > 0:
                    self.btn_prev_err.setEnabled(True)
                    self.btn_next_err.setEnabled(True)
                    self.set_frame_position(self.logic.error_frames_list[0])
            except Exception as e:
                QMessageBox.critical(self, "Error", f"[ERROR] Failed to process {filename}:\n{str(e)}")

    #user interface actions
    def undo_action(self):
        if self.logic.undo(): self.update_gui_after_action()

    def redo_action(self):
        if self.logic.redo(): self.update_gui_after_action()

    def reject_all_deductions(self):
        if self.logic.reject_all_in_frame(self.current_frame):
            self.update_gui_after_action()

    def update_gui_after_action(self):
        self.score.setText(f"{max(0, self.logic.e_score):.1f}")
        self.update_undo_redo_buttons()
        self.update_log_for_current_frame()
        self.refresh_display()

    def update_undo_redo_buttons(self):
        self.btn_undo.setEnabled(len(self.logic.undo_stack) > 0)
        self.btn_redo.setEnabled(len(self.logic.redo_stack) > 0)

    #playback
    def toggle_playback(self):
        if self.is_playing:
            self.timer.stop()
            self.btn_play_pause.setIcon(QIcon("gui/assets/play.png"))
        else:
            fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 30
            self.timer.start(int(1000 / fps))
            self.btn_play_pause.setIcon(QIcon("gui/assets/pause.png"))
        self.is_playing = not self.is_playing

    def next_frame(self):
        if not self.video_cap: return
        if self.current_frame >= int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1:
            self.toggle_playback()
            return
            
        ret, frame = self.video_cap.read()
        if ret:
            self.current_frame += 1
            self.frame_slider.setValue(self.current_frame)
            self.display_frame(frame, self.current_frame)
            self.update_log_for_current_frame()

    def set_frame_position(self, frame_idx):
        if self.video_cap:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.video_cap.read()
            if ret:
                self.current_frame = frame_idx
                self.frame_slider.setValue(frame_idx)
                self.display_frame(frame, frame_idx)
                self.update_log_for_current_frame()

    def jump_prev_error(self):
        prev_errs = [f for f in self.logic.error_frames_list if f < self.current_frame]
        if prev_errs: self.set_frame_position(max(prev_errs))

    def jump_next_error(self):
        next_errs = [f for f in self.logic.error_frames_list if f > self.current_frame]
        if next_errs: self.set_frame_position(min(next_errs))

    #frame displaying and log updating
    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None: widget.deleteLater()

    def update_log_for_current_frame(self):
        self.clear_layout(self.log_layout)
        
        if self.current_frame in self.logic.errors_by_frame:
            data = self.logic.errors_by_frame[self.current_frame]
            self.lbl_acrobatic_info.setStyleSheet(css.ACROBATIC_INFO_STYLE)
            self.lbl_confidence_info.setStyleSheet(css.CONFIDENCE_INFO_STYLE)

            if data["acrobatic"] == "Transition":
                self.btn_reject_all.setEnabled(False)
                self.lbl_acrobatic_info.setText("<b>Detected Acrobatic:</b>TRANSITION")
            else:
                self.btn_reject_all.setEnabled(True)
                self.lbl_acrobatic_info.setText(f"<b>Detected Acrobatic:</b> {data['acrobatic'].upper()}")
                
            self.lbl_confidence_info.setText(f"<b>Confidence:</b> {data['confidence']:.2f}%")
            
            if not data["reasons"]:
                lbl = QLabel("Perfect execution. No deductions.")
                lbl.setStyleSheet(css.LBL_REASON_STYLE)
                self.log_layout.addWidget(lbl)
            else:
                for i, reason_obj in enumerate(data["reasons"]):
                    self.log_layout.addWidget(DeductionWidget(self.current_frame, i, reason_obj, self))
        else:
            self.btn_reject_all.setEnabled(False)
            self.lbl_acrobatic_info.setText("<b>Detected Acrobatic:</b> None")
            self.lbl_confidence_info.setText("<b>Confidence:</b> None")
            lbl = QLabel("No data available for this frame.")
            lbl.setStyleSheet("color: gray;")
            self.log_layout.addWidget(lbl)
            
        self.log_layout.addStretch()

    def display_frame(self, frame, frame_idx):
        if self.current_view_mode == "Skeleton" and frame_idx in self.logic.errors_by_frame:
            data = self.logic.errors_by_frame[frame_idx]
            active_breakdowns = [r["text"] for r in data["reasons"] if r["status"] != "rejected"]
            is_false_pos = (data["acrobatic"] == "Transition")
            display_img = generate_skeleton_canvas(data["position"], active_breakdowns, is_false_pos)
        else:
            display_img = frame
            
        img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        qt_img = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format.Format_RGB888)
        self.video.setPixmap(QPixmap.fromImage(qt_img).scaled(self.video.size(), Qt.AspectRatioMode.KeepAspectRatio))
        
        total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.video_cap else 0
        self.lbl_frame_counter.setText(f"Frame: {frame_idx}/{total_frames}")

    def refresh_display(self):
        if self.video_cap: self.set_frame_position(self.current_frame)
            
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_display()