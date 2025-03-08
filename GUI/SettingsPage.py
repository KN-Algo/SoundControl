from BasePage import BasePage


class SettingsPage(BasePage):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SettingsPage")

        self.btn_back = self.create_back_button(clicked_callback=self.back_clicked)

    def back_clicked(self):
        self.switch_page(1)  # Powrót do MainPage
