from BasePage import BasePage


class MainPage(BasePage):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("MainPage")

        center_x = (1920 - self.BUTTON_WIDTH) // 2
        first_button_y = 400
        second_button_y = 550

        self.btn_select_source = self.create_button(
            "Select Sound Source",
            (center_x, first_button_y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT),
            clicked_callback=self.select_source_clicked,
        )
        self.btn_settings = self.create_button(
            "Settings",
            (center_x, second_button_y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT),
            clicked_callback=self.settings_clicked,
        )

    def select_source_clicked(self):
        print("Select Sound Source clicked!")
        self.switch_page(3)  # SoundSelectionPage - indeks 3

    def settings_clicked(self):
        print("Settings clicked!")
        self.switch_page(2)  # SettingsPage - indeks 2
