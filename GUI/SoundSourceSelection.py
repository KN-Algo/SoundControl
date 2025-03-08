from BasePage import BasePage


class SoundSelectionPage(BasePage):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SoundSelectionPage")
        # Ustalanie parametrów rozmieszczenia przycisków
        start_x = 560
        start_y = 300
        col_spacing = 400
        row_spacing = 150
        sound_names = ["Pianino", "Gitara", "Flet", "Trąbka", "Głos", "Kontrabas"]

        for i, name in enumerate(sound_names):
            row = i // 2
            col = i % 2
            x = start_x + col * col_spacing
            y = start_y + row * row_spacing
            btn = self.create_button(
                name,
                (x, y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT),
                clicked_callback=self.sound_selected,
            )

        self.btn_back = self.create_back_button(clicked_callback=self.back_clicked)

    def back_clicked(self):
        self.switch_page(1)

    def sound_selected(self):
        self.switch_page(4)
