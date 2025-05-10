from PyQt5.QtWidgets import (
    QWidget,
    QPushButton,
    QInputDialog,
    QVBoxLayout,
    QGridLayout,
    QSizePolicy,
    QScrollArea,
    QSpacerItem,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from BasePage import BasePage


class SettingsPage(BasePage):

    NOTES = [
        f"{note}{octave}"
        for octave in range(0, 9)
        for note in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    ]
    NOTES = NOTES[9:-3]  # Od A0 do C8

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SettingsPage")

        self.key_bindings = {}  # np. {'A0': 'a'}
        self.note_buttons = {}  # np. {'A0': QPushButton}

        # Przyciski + layout
        self.init_ui()

    def init_ui(self):
        # Powrót
        self.btn_back = self.create_back_button(clicked_callback=self.back_clicked)

        # Główny layout z przewijaniem
        scroll_area = QScrollArea(self)
        scroll_area.setGeometry(100, 100, 1720, 880)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("background-color: transparent; border: none;")

        scroll_widget = QWidget()
        scroll_area.setWidget(scroll_widget)

        layout = QVBoxLayout(scroll_widget)
        layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        # Siatka przycisków
        grid = QGridLayout()
        grid.setSpacing(20)

        buttons_per_row = 8
        for i, note in enumerate(self.NOTES):
            row = i // buttons_per_row
            col = i % buttons_per_row

            btn = QPushButton(f"{note} : ...")
            btn.setFont(QFont("Noto Sans", 14))
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            btn.setMinimumHeight(60)

            btn.setStyleSheet(
                """
                QPushButton {
                    background-color: lightgray;
                    color: black;
                    border-radius: 10px;
                }
                QPushButton:hover {
                    background-color: silver;
                }
            """
            )
            btn.clicked.connect(lambda _, n=note: self.assign_key(n))

            self.note_buttons[note] = btn
            grid.addWidget(btn, row, col)

        layout.addSpacerItem(
            QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )
        layout.addLayout(grid)

    def back_clicked(self):
        self.switch_page(1)  # Powrót do MainPage

    def assign_key(self, note):
        # Ustawiamy styl dialogu
        style = """
            QInputDialog {
                background-color: lightgray;
            }
            QLabel {
                background-color: lightgray;
                color: black;
                font-size: 14px;
            }
            QLineEdit {
                background-color: white;
                color: black;
                font-size: 14px;
            }
            QPushButton {
                background-color: silver;
                color: black;
            }
            QPushButton:hover {
                background-color: gray;
            }
        """

        # Tworzymy dialog ręcznie, by przypisać styl
        dlg = QInputDialog(self)
        dlg.setStyleSheet(style)
        dlg.setInputMode(QInputDialog.TextInput)
        dlg.setLabelText(f"Podaj klawisz dla {note}:")
        dlg.setWindowTitle("Przypisz klawisz")

        if dlg.exec_() == QInputDialog.Accepted:
            key = dlg.textValue()
            if key:
                self.key_bindings[note] = key
                btn = self.note_buttons.get(note)
                if btn:
                    btn.setText(f"{note} : {key}")
                print(f"Przypisano {key} do {note}")
