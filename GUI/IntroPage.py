from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QPoint
from PyQt5.QtGui import QFont
from BasePage import BasePage


class IntroPage(BasePage):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("IntroPage")
        # Nadpisujemy ewentualne ustawienia, zachowując tło z BasePage

        # ----- Etykieta "Sound Control" -----
        self.sound_control_label = QLabel("Sound Control", self)
        self.sound_control_label.setAlignment(Qt.AlignCenter)
        label_font = QFont("Times New Roman", 48, QFont.Bold)
        self.sound_control_label.setFont(label_font)
        self.sound_control_label.setStyleSheet(
            "color: rgb(107, 61, 216); background: transparent;"
        )
        self.sound_control_label.adjustSize()

        # ----- Etykieta "Press here to start" -----
        self.press_label = QLabel("Press here to start", self)
        self.press_label.setAlignment(Qt.AlignCenter)
        press_font = QFont("Times New Roman", 24, QFont.Bold)
        self.press_label.setFont(press_font)
        self.press_label.setStyleSheet(
            "color: rgb(107,61,216); background: transparent;"
        )
        self.press_label.adjustSize()
        self.press_label.hide()

        # Animacja
        self.animation_done = False
        self.animation = None

    def showEvent(self, event):
        """Uruchamiamy animację przy pierwszym wyświetleniu."""
        super().showEvent(event)
        if self.animation is None:
            self.init_animation()

    def init_animation(self):
        offset_right = 30
        # Pozycja startowa
        start_x = (self.width() - self.sound_control_label.width()) // 2 + offset_right
        start_y = self.height()
        self.sound_control_label.move(start_x, start_y)

        self.animation = QPropertyAnimation(self.sound_control_label, b"pos", self)
        self.animation.setDuration(2000)
        self.animation.setEasingCurve(QEasingCurve.OutBounce)
        self.animation.setStartValue(QPoint(start_x, start_y))

        final_x = (self.width() - self.sound_control_label.width()) // 2 + offset_right
        final_y = int(self.height() * 0.5) - (self.sound_control_label.height() // 2)
        self.animation.setEndValue(QPoint(final_x, final_y))

        self.animation.finished.connect(self.show_press_label)
        self.animation.start()

    def show_press_label(self):
        """Pokazujemy napis 'Press here to start'."""
        self.animation_done = True
        self.reposition_press_label()
        self.press_label.show()

    def reposition_press_label(self):
        offset_right = 30
        sc_label_bottom = (
            self.sound_control_label.y() + self.sound_control_label.height()
        )
        space_to_bottom = self.height() - sc_label_bottom

        new_y = (
            sc_label_bottom + (space_to_bottom // 2) - (self.press_label.height() // 2)
        )
        new_x = (self.width() - self.press_label.width()) // 2 + offset_right
        self.press_label.move(new_x, new_y)

    def mousePressEvent(self, event):
        """Kliknięcie w widok przenosi do MainPage."""
        print("IntroPage clicked!")
        self.switch_page(1)

    def resizeEvent(self, event):
        """Repozycjonowanie napisów po zmianie rozmiaru."""
        super().resizeEvent(event)
        if (
            self.animation
            and self.animation.state() == QPropertyAnimation.Stopped
            and self.animation_done
        ):
            self.reposition_final_labels()

    def reposition_final_labels(self):
        offset_right = 30
        final_x = (self.width() - self.sound_control_label.width()) // 2 + offset_right
        final_y = int(self.height() * 0.5) - (self.sound_control_label.height() // 2)
        self.sound_control_label.move(final_x, final_y)
        if self.press_label.isVisible():
            self.reposition_press_label()
