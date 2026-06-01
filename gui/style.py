MAIN_BG_STYLE = "background-color: #e5e7eb;" 
LEFT_COL_STYLE = "background-color: #f3f4f6; border-radius: 12px;" 
RIGHT_COL_STYLE = "background-color: #f3f4f6; border-radius: 12px;"

#left column
TAB_STYLE = """
    QPushButton { 
        background-color: #e0e0e0; 
        color: #888;
        padding: 12px; 
        font-size: 15px;
        border: none; 
        border-radius: 12px;
        font-weight: bold; 
        text-align: center;
    }
    QPushButton:hover:!checked {
        background-color: #d5d5d5;
    }
    QPushButton:checked { 
        background-color: #2196F3; 
        color: white; 
    }
"""

VIDEO_STYLE = "background-color: #000; color: white; font-size: 16px; border-radius: 12px;"

#right column
TITLE_STYLE = "font-size: 24px; color: #333; padding: 5px;"

LOG_CONTAINER_STYLE = "background-color: #f9f9f9; padding: 5px;" 
SCROLL_AREA_STYLE = "QScrollArea {background-color: #f9f9f9; border-radius: 12px; }"

ACROBATIC_INFO_STYLE = "font-size: 16px; color: #666; font-weight: bold; padding: 5px;"
CONFIDENCE_INFO_STYLE = "font-size: 14px; color: #999; font-weight: bold; padding: 5px;"

LBL_REASON_STYLE = "font-size: 14px; border: none; padding: 5px;"

SCORETITLE_STYLE = "font-size: 18px; font-weight: bold; color: #aaa; padding: 5px; margin-top: 10px;"
SCORE_STYLE = "font-size: 32px; font-weight: bold; color: #777; padding: 5px; border-radius: 12px;"

BTN_REJECT_ALL_STYLE = """
    QPushButton {
        background-color: #2196F3; 
        color: white; 
        font-weight: bold; 
        padding: 10px; 
        margin-top: 5px; 
        border-radius: 8px;
        border: none;
    }
    QPushButton:hover {
        background-color: #1976D2;
    }
"""

BTN_ACC_STYLE = """
    QPushButton {
        background-color: #4CAF50; 
        color: white; 
        font-weight: bold; 
        padding: 10px;
        border-radius: 12px;
        border: none;
    }
    QPushButton:hover {
        background-color: #45a049;
    }
"""

BTN_REJ_STYLE = """
    QPushButton {
        background-color: #F44336; 
        color: white; 
        font-weight: bold; 
        padding: 10px;
        border-radius: 12px;
        border: none;
    }
    QPushButton:hover {
        background-color: #d32f2f;
    }
"""

BTN_EXPORT_STYLE = """
    QPushButton {
        background-color: #4B5563; 
        color: white; 
        font-weight: bold; 
        padding: 10px; 
        margin-top: 5px; 
        border-radius: 8px;
        border: none;
    }
    QPushButton:hover {
        background-color: #374151;
    }
"""

DEDUCTION_ACCEPTED = "background-color: #d4edda; color: #155724; border-radius: 12px; margin-bottom: 5px;"
DEDUCTION_REJECTED = "background-color: #e2e3e5; color: #383d41; border-radius: 12px; margin-bottom: 5px;"
DEDUCTION_PENDING = "background-color: white; color: black; border-radius: 12px; margin-bottom: 5px;"

SCROLLBAR_STYLE = """
    QScrollBar:vertical {
        border: none;
        background: #f1f1f1;
        width: 10px;
        margin: 0px 0px 0px 0px;
        border-radius: 5px;
    }
    QScrollBar::handle:vertical {
        background: #c1c1c1;
        min-height: 20px;
        border-radius: 5px;
    }
    QScrollBar::handle:vertical:hover {
        background: #a8a8a8;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
        border: none;
        background: none;
    }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
        background: none;
    }
"""