import sys

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QStackedWidget,
)

from IntroPage import IntroPage
from MainPage import MainPage
from SettingsPage import SettingsPage
from SoundSourceSelection import SoundSelectionPage
from LivePage import LiveFrequencyPage


class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sound Control")
        self.setGeometry(100, 100, 1920, 1080)
        self.stacked_widget = QStackedWidget()

        self.intro_page = IntroPage(parent=self)
        self.main_page = MainPage(parent=self)
        self.settings_page = SettingsPage(parent=self)
        self.sound_selection_page = SoundSelectionPage(parent=self)
        self.live_frequency_page = LiveFrequencyPage(parent=self)

        # Dodajemy strony do QStackedWidget według indeksów:
        self.stacked_widget.addWidget(self.intro_page)
        self.stacked_widget.addWidget(self.main_page)
        self.stacked_widget.addWidget(self.settings_page)
        self.stacked_widget.addWidget(self.sound_selection_page)
        self.stacked_widget.addWidget(self.live_frequency_page)

        self.setCentralWidget(self.stacked_widget)
        self.stacked_widget.setCurrentIndex(0)

    def switch_page(self, index):
        """Zmienia stronę w QStackedWidget na podany indeks."""
        self.stacked_widget.setCurrentIndex(index)


def main():
    app = QApplication(sys.argv)
    window = AppWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
