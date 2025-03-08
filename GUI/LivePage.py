from PyQt5.QtWidgets import QLabel
from BasePage import BasePage


class LiveFrequencyPage(BasePage):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("FrequencyDisplayPage")
        # Ekran wyświetlania częstotliwości
        self.freq_display = QLabel(self)
        self.freq_display.setStyleSheet("background-color: rgb(0, 0, 0);")
        freq_width = int(1920 * 0.4)
        freq_height = int(1080 * 0.4)
        self.freq_display.setFixedSize(freq_width, freq_height)
        self.freq_display.move(
            (1920 - freq_width) // 2,
            (1080 - freq_height) // 2,
        )

        # Przycisk "Start"
        # pozycjonowany na środku freq_display
        start_x = (
            self.freq_display.x() + (self.freq_display.width() - self.BUTTON_WIDTH) // 2
        )
        start_y = (
            self.freq_display.y()
            + (self.freq_display.height() - self.BUTTON_HEIGHT) // 2
        )
        self.start_button = self.create_button(
            "Start", (start_x, start_y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT)
        )

        self.btn_back = self.create_back_button(clicked_callback=self.back_clicked)

    def back_clicked(self):
        self.switch_page(3)
