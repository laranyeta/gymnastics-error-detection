import os
import cv2
import gui.style as css

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QSlider, QPushButton, QFileDialog, 
                             QMessageBox, QScrollArea)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QImage, QPixmap, QAction, QIcon, QKeySequence, QShortcut
from PyQt6.QtGui import QTextDocument, QPageLayout, QPageSize, QPdfWriter
from PyQt6.QtCore import QMarginsF

from backend.rnn.predict import resource_path
from backend.rnn.score import generate_skeleton_canvas
from gui.components import DeductionWidget
from gui.logic import AppLogic

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gymnastics Error Detector")
        self.resize(1200, 750)
        self.logic = AppLogic()
        self.video_cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.is_playing = False
        self.current_frame = 0  
        self.current_view_mode = "Video"
        self.setAcceptDrops(True)
        
        self.create_menu_bar()
        self.setup_ui()

    def create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')

        load_video_action = QAction('Load video', self)
        load_video_action.setShortcut(QKeySequence("CTRL+O")) #open shortcut
        load_video_action.triggered.connect(self.load_video)
        file_menu.addAction(load_video_action)

        self.load_json_action = QAction('Load JSON data', self)
        self.load_json_action.triggered.connect(self.load_json)
        self.load_json_action.setEnabled(False) 
        file_menu.addAction(self.load_json_action)

        export_report_action = QAction('Export report', self)
        export_report_action.setShortcut(QKeySequence("CTRL+S")) #open shortcut
        export_report_action.triggered.connect(self.export_report)
        file_menu.addAction(export_report_action)

    def setup_ui(self):
        central_widget = QWidget()
        central_widget.setStyleSheet(css.MAIN_BG_STYLE)
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15) #space between columns

        #left container
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
        
        self.video = QLabel("Drag and drop a video here or use File > Load Video to get started")
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
        
        self.btn_prev_err = create_icon_button(resource_path("gui/assets/prev.png"), self.jump_prev_error)
        self.btn_prev_err.setEnabled(False)
        
        self.btn_play_pause = create_icon_button(resource_path("gui/assets/play.png"), self.toggle_playback)
        self.btn_play_pause.setEnabled(False)
        
        self.btn_next_err = create_icon_button(resource_path("gui/assets/next.png"), self.jump_next_error)
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
        
        #right container
        self.right_container = QWidget()
        self.right_container.setStyleSheet(css.RIGHT_COL_STYLE)
        col_R = QVBoxLayout(self.right_container)
        
        logs_header_layout = QHBoxLayout()
        self.lbl_logs_title = QLabel("<b>Deductions Logs</b>")
        self.lbl_logs_title.setStyleSheet(css.TITLE_STYLE)
        
        self.btn_undo = create_icon_button(resource_path("gui/assets/undo.png"), self.undo_action)
        self.btn_redo = create_icon_button(resource_path("gui/assets/redo.png"), self.redo_action)
        
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
        
        buttons_action_layout = QHBoxLayout()
        buttons_action_layout.setSpacing(10)

        self.btn_reject_all = QPushButton("False Positive (Discard frame)") #marks as transition
        self.btn_reject_all.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reject_all.setStyleSheet(css.BTN_REJECT_ALL_STYLE)
        self.btn_reject_all.clicked.connect(self.reject_all_deductions)

        buttons_action_layout.addWidget(self.btn_reject_all, stretch=1)

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
        col_R.addLayout(buttons_action_layout)
        col_R.addWidget(self.score_title)
        col_R.addWidget(self.score)

        main_layout.addWidget(left_container, stretch=5)
        main_layout.addWidget(self.right_container, stretch=5)

        self.lbl_acrobatic_info.hide()
        self.lbl_confidence_info.hide()
        self.btn_reject_all.hide()
        self.btn_undo.hide()
        self.btn_redo.hide()
        self.update_undo_redo_buttons()

        #undo-redo shortcuts
        self.undo_shortcut = QShortcut(QKeySequence("CTRL+Z"), self)
        self.undo_shortcut.activated.connect(self.undo_action)
        self.redo_shortcut = QShortcut(QKeySequence("CTRL+Y"), self)
        self.redo_shortcut.activated.connect(self.redo_action)

        #false positive
        self.undo_shortcut = QShortcut(QKeySequence("CTRL+F"), self)
        self.undo_shortcut.activated.connect(self.reject_all_deductions)

        #export report
        self.redo_shortcut = QShortcut(QKeySequence("CTRL+S"), self)
        self.redo_shortcut.activated.connect(self.export_report)

        #play/pause
        self.space_shortcut = QShortcut(QKeySequence("Space"), self)
        self.space_shortcut.activated.connect(self.toggle_playback)

        #skip frames
        self.next_frame_shortcut = QShortcut(QKeySequence("Right"), self)
        self.next_frame_shortcut.activated.connect(self.step_forward_frame)
        self.prev_frame_shortcut = QShortcut(QKeySequence("Left"), self)
        self.prev_frame_shortcut.activated.connect(self.step_backward_frame)

        #next/prev error
        self.next_err_shortcut = QShortcut(QKeySequence("X"), self)
        self.next_err_shortcut.activated.connect(self.jump_next_error)
        self.prev_err_shortcut = QShortcut(QKeySequence("Z"), self)
        self.prev_err_shortcut.activated.connect(self.jump_prev_error)

    def switch_view_mode(self, mode):
        self.current_view_mode = mode
        self.btn_mode_video.setChecked(mode == "Video")
        self.btn_mode_skeleton.setChecked(mode == "Skeleton")
        self.refresh_display()

    #load video + searches for json in same directory
    def load_video(self, filepath=None):
        if filepath:
            filename = filepath
        else:
            filename, _ = QFileDialog.getOpenFileName(self, "Select video", "", "videos (*.mp4 *.avi)")
        if filename:
            if self.video_cap: self.video_cap.release()
            self.video_cap = cv2.VideoCapture(filename)
            self.frame_slider.setMaximum(int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
            self.frame_slider.setEnabled(True)
            self.btn_play_pause.setEnabled(True)
            self.set_frame_position(0)
            
            dir_name = os.path.dirname(filename)
            base_name = os.path.basename(filename)
            
            name_without_ext = os.path.splitext(base_name)[0]
            if name_without_ext.endswith("_skeleton"): #test01_skeleton.mp4 -> test01.json
                name_without_ext = name_without_ext.replace("_skeleton", "") #deletes _skeleton from filename
                
            json_filename = os.path.join(dir_name, f"{name_without_ext}.json")
            if os.path.exists(json_filename):
                try:
                    total_errors = self.logic.load_json_data(json_filename)
                    self.update_gui_after_action()
                
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
                        
                    self.load_json_action.setEnabled(False)
                    
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"[ERROR] Failed to auto-load {json_filename}:\n{str(e)}")
                    self.load_json_action.setEnabled(True)
            else:
                QMessageBox.information(self, "Info", f"[WARNING] Video was loaded  but couldn't find the associated data file ({name_without_ext}.json).\n\nYou can load it manually via 'File > Load Data'.")
                self.load_json_action.setEnabled(True)
            
    def load_json(self): #load json manually in case it wasn't auto-loaded
        filename, _ = QFileDialog.getOpenFileName(self, "Select JSON", "", "JSON (*.json)")
        if filename:
            try:
                total_errors = self.logic.load_json_data(filename)
                self.update_gui_after_action()
                
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

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            filepath = url.toLocalFile()
            if filepath.lower().endswith(('.mp4', '.avi')):
                self.load_video(filepath)
                break #only loads first video

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
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) #check if we are at the end of the video, if so start from the beginning
            if self.current_frame >= total_frames - 1:
                self.set_frame_position(0)

            fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 30
            self.timer.start(int(1000 / fps))
            self.btn_play_pause.setIcon(QIcon(resource_path("gui/assets/pause.png"))) #show pause icon
            self.right_container.hide() #only shows video when playing
            
        else:
            self.timer.stop()
            self.btn_play_pause.setIcon(QIcon(resource_path("gui/assets/play.png"))) #show play icon
            self.right_container.show() #show logs when paused
            self.update_log_for_current_frame()

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

    def step_forward_frame(self):
        if self.video_cap:
            total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.current_frame < total_frames - 1:
                self.set_frame_position(self.current_frame + 1)

    def step_backward_frame(self):
        if self.video_cap and self.current_frame > 0:
            self.set_frame_position(self.current_frame - 1)
        
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
        if self.current_frame in self.logic.active_acrobatic: 
            acro_data = self.logic.active_acrobatic[self.current_frame]
            acrobatic_name = acro_data["acrobatic"]
            confidence = acro_data["confidence"]
            
            if self.current_frame in self.logic.errors_by_frame:
                acrobatic_name = self.logic.errors_by_frame[self.current_frame]["acrobatic"]

            self.lbl_acrobatic_info.setStyleSheet(css.ACROBATIC_INFO_STYLE)
            self.lbl_confidence_info.setStyleSheet(css.CONFIDENCE_INFO_STYLE)

            if acrobatic_name == "Transition":
                self.lbl_acrobatic_info.setText("<b>Detected Acrobatic:</b> TRANSITION") #mark as transition
            else:
                self.lbl_acrobatic_info.setText(f"<b>Detected Acrobatic:</b> {acrobatic_name.upper()}")
                
            self.lbl_confidence_info.setText(f"<b>Confidence:</b> {confidence:.2f}%")
        else:
            self.lbl_acrobatic_info.setText("<b>Detected Acrobatic:</b> None")
            self.lbl_confidence_info.setText("<b>Confidence:</b> None")

        if self.current_frame in self.logic.errors_by_frame:
            data = self.logic.errors_by_frame[self.current_frame]
            
            if data["acrobatic"] == "Transition":
                self.btn_reject_all.setEnabled(False)
            else:
                self.btn_reject_all.setEnabled(True)
            
            if not data["reasons"]:
                lbl = QLabel("Perfect execution. No deductions.")
                lbl.setStyleSheet(css.LBL_REASON_STYLE)
                self.log_layout.addWidget(lbl)
            else:
                for i, reason_obj in enumerate(data["reasons"]):
                    self.log_layout.addWidget(DeductionWidget(self.current_frame, i, reason_obj, self))
        else:
            self.btn_reject_all.setEnabled(False)
            lbl = QLabel("Possible deductions will appear during peak frame of the execution.")
            lbl.setStyleSheet("color: gray; font-style: italic; padding: 10px;")
            self.log_layout.addWidget(lbl)
            
        self.log_layout.addStretch()

    def display_frame(self, frame, frame_idx):
        if self.current_view_mode == "Skeleton": #skeleton mode
            if frame_idx in self.logic.errors_by_frame: #frame with error -> paint skeleton with deductions highlighted
                data = self.logic.errors_by_frame[frame_idx]
                active_breakdowns = [r["text"] for r in data["reasons"] if r["status"] != "rejected"]
                is_false_pos = (data["acrobatic"] == "Transition")
                display_img = generate_skeleton_canvas(data["position"], active_breakdowns, is_false_pos)
            
            else: #frame without error -> paint skeleton without deductions
                try:
                    raw_frame_data = self.logic.raw_data[frame_idx] 
                    position = raw_frame_data["position"]
                    display_img = generate_skeleton_canvas(position, [], False)
                except Exception:
                    display_img = frame
        else: #video mode
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
    
    def export_report(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save PDF Report", "", "PDF Files (*.pdf);;All Files (*)")
        
        if filename:
            if not filename.endswith('.pdf'):
                filename += '.pdf'
                
            html_content = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #2c3e50; margin: 30px; }}
                    h1 {{ color: #1a252f; border-bottom: 2px solid #bdc3c7; padding-bottom: 8px; font-size: 24px; }}
                    h3 {{ color: #34495e; margin-top: 25px; font-size: 16px; border-bottom: 1px solid #ecf0f1; padding-bottom: 5px; }}
                    
                    .score-box {{ background-color: #fff; border-left: 4px solid #3498db; padding: 12px; margin: 20px 0; border-radius: 0 6px 6px 0; }}
                    .score-title {{ font-size: 11px; color: #57606f; font-weight: bold; letter-spacing: 1px; }}
                    .score-value {{ font-size: 26px; font-weight: bold; color: #2c3e50; margin-top: 5px; }}
                    
                    .frame-title {{ font-size: 13px; font-weight: bold; color: #2c3e50; margin-top: 15px; background-color: #f0f3f4; padding: 12px 16px; border-radius: 8px; }}
                    .deduction-table {{ width: 100%; border-collapse: collapse; margin-top: 5px; }}
                    .deduction-row {{ border-bottom: 1px solid #f1f2f6; }}
                    .deduction-text {{ font-size: 12px; color: #57606f; padding: 10px 4px; }}
                    
                    .badge {{ font-size: 10px; font-weight: bold; color: white; padding: 6px 12px; border-radius: 6px; text-align: center; display: inline-block; }}
                    .badge-minor {{ background-color: #2ecc71; }}
                    .badge-medium {{ background-color: #e67e22; }}
                    .badge-severe {{ background-color: #e74c3c; }}
                    .badge-accepted {{ background-color: #81c784; padding: 6px 14px; border-radius: 8px; }}
                    .badge-rejected {{ background-color: #e57373; padding: 6px 14px; border-radius: 8px; }}
                </style>
            </head>
            <body>
                <h1>Gymnastics Performance Report</h1>
                
                <div class="score-box">
                    <div class="score-title">FINAL E-SCORE</div>
                    <div class="score-value">{self.logic.e_score:.1f}</div>
                </div>
                
                <h3>Deductions Breakdown</h3>
            """
            
            if not self.logic.errors_by_frame:
                html_content += "<p style='color: #7f8c8d; font-style: italic; font-size: 12px;'>No acrobatics recorded.</p>"
            else:
                for frame, data in self.logic.errors_by_frame.items():
                    html_content += f"""
                    <div class="frame-title">Frame {frame} | {data['acrobatic'].upper()}</div>
                    """
                    
                    reasons = data.get("reasons", [])
                    if not reasons:
                        html_content += "<p style='color: #7f8c8d; font-style: italic; font-size: 12px; margin: 10px 0 10px 5px;'>Perfect execution. No deductions.</p>"
                    
                    else:
                        html_content += '<table class="deduction-table">'
                        for r in reasons:
                            status = r['status'].upper()
                            
                            if status == "SEVERE":
                                badge_class = "badge-severe"
                            elif status == "MEDIUM":
                                badge_class = "badge-medium"
                            elif status == "MINOR":
                                badge_class = "badge-minor"
                            elif status == "ACCEPTED":
                                badge_class = "badge-accepted"
                            elif status == "REJECTED":
                                badge_class = "badge-rejected"
                            else:
                                badge_class = "badge-minor"
                                
                            html_content += f"""
                            <tr class="deduction-row">
                                <td style="width: 95px; padding: 10px 0;">
                                    <span class="badge {badge_class}">{status}</span>
                                </td>
                                <td class="deduction-text">{r['text']}</td>
                            </tr>
                            """
                        html_content += "</table>"
                    
            html_content += """
            </body>
            </html>
            """
            document = QTextDocument()
            document.setHtml(html_content)
            
            writer = QPdfWriter(filename)
            writer.setResolution(96)
            layout = QPageLayout()
            layout.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
            layout.setOrientation(QPageLayout.Orientation.Portrait)
            layout.setMargins(QMarginsF(5, 5, 5, 5))
            writer.setPageLayout(layout)

            document.print(writer)            
            QMessageBox.information(self, "Success", "PDF report exported successfully!")