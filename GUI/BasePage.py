from PyQt5.QtWidgets import QWidget, QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class BasePage(QWidget):
    """
    Klasa bazowa ustawia:
      - domyślne tło (BACKGROUND_COLOR)
      - domyślną czcionkę "Noto Sans"
      - domyślne parametry przycisków: kolor tekstu (BUTTON_TEXT_COLOR), tło,
        border-radius, rozmiar, efekt hover
      - metody pomocnicze do tworzenia przycisków (create_button, create_back_button)
      - mechanizm zmiany strony oparty na indeksowaniu (switch_page)
    """

    BACKGROUND_COLOR = "rgb(43, 45, 48)"
    BUTTON_TEXT_COLOR = "rgb(107, 61, 216)"
    BUTTON_BG_COLOR = "darkgray"
    BUTTON_HOVER_BG_COLOR = "rgb(3,3,3)"
    BUTTON_BORDER_RADIUS = "20px"
    BUTTON_FONT = QFont("Noto Sans", 20, QFont.Bold)
    BUTTON_WIDTH = 350
    BUTTON_HEIGHT = 80

    BACK_BUTTON_WIDTH = 60
    BACK_BUTTON_HEIGHT = 60
    BACK_BUTTON_FONT = QFont("Noto Sans", 25, QFont.Bold)
    BACK_BUTTON_STYLE = f"""
        QPushButton {{
            background-color: rgba(11, 12, 17, 0);
            color: {BUTTON_TEXT_COLOR};
            border-radius: 30px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: rgb(0, 0, 0);
        }}
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMinimumSize(1920, 1080)
        self.setStyleSheet(f"background-color: {self.BACKGROUND_COLOR};")
        self.setFont(QFont("Noto Sans", 10))

    def create_button(self, text, geometry, font=None, clicked_callback=None):
        """
        Tworzy przycisk z domyślnymi ustawieniami:
          - tło, kolor tekstu, border-radius, efekt hover
          - wykorzystuje BUTTON_FONT, BUTTON_WIDTH i BUTTON_HEIGHT zdefiniowane w BasePage
          - geometry to krotka (x, y, width, height)
        """
        button = QPushButton(text, self)
        if font is None:
            font = self.BUTTON_FONT
        button.setFont(font)
        button.setGeometry(*geometry)
        button_style = f"""
            QPushButton {{
                background-color: {self.BUTTON_BG_COLOR};
                color: {self.BUTTON_TEXT_COLOR};
                border-radius: {self.BUTTON_BORDER_RADIUS};
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.BUTTON_HOVER_BG_COLOR};
            }}
        """
        button.setStyleSheet(button_style)
        if clicked_callback is not None:
            button.clicked.connect(clicked_callback)
        return button

    def create_back_button(self, clicked_callback=None):
        """
        Tworzy przycisk "Powrót" korzystając z BACK_BUTTON_STYLE.
        Po kliknięciu wywołuje przekazany callback.
        """
        btn = QPushButton("←", self)
        btn.setFont(self.BACK_BUTTON_FONT)
        btn.setGeometry(30, 30, self.BACK_BUTTON_WIDTH, self.BACK_BUTTON_HEIGHT)
        btn.setStyleSheet(self.BACK_BUTTON_STYLE)
        if clicked_callback is not None:
            btn.clicked.connect(clicked_callback)
        return btn

    def switch_page(self, index):
        main_window = self.window()
        if hasattr(main_window, "switch_page"):
            main_window.switch_page(index)
