import argparse
import yaml
import numpy as np
import pyaudio
from ctypes import c_uint, windll
from time import time
import os

# Opcjonalny wykres
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parametry audio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Stałe WinAPI
KEYEVENTF_UP = 0x0002
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true", help="Włącz wykres podglądu")
parser.add_argument("--profile", default="default", help="Nazwa profilu mapowania nut (bez .yaml)")
parser.add_argument("--settings", default="settings.yaml", help="Ścieżka do pliku ustawień systemowych")
parser.add_argument("--mic", action="store_true", help="Wylistuj dostępne mikrofony i ustaw jeden z nich")

args = parser.parse_args()

# Wczytaj ustawienia systemowe
with open(args.settings, "r") as f:
    settings = yaml.safe_load(f)

AMPLITUDE_THRESHOLD = settings.get("amplitude_threshold", 1000)
TRIGGER_COOLDOWN = settings.get("trigger_cooldown", 0.4)
SENSITIVITY = settings.get("sensitivity", 30)
INPUT_DEVICE_NAME = settings.get("input_device", None)

# Inicjalizacja PyAudio
p = pyaudio.PyAudio()

# Obsługa przełącznika --mic
if args.mic:
    print("Dostępne mikrofony:")
    devices = []
    seen_names = set()  # Zbiór do przechowywania unikalnych nazw urządzeń
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:  # Filtruj tylko urządzenia wejściowe
            if info["name"] not in seen_names:  # Sprawdź, czy nazwa już była
                seen_names.add(info["name"])
                devices.append((i, info["name"]))
                print(f"{len(devices)}: {info['name']}")

    if not devices:
        print("Nie znaleziono żadnych urządzeń wejściowych.")
        exit(1)

    choice = input("Wybierz numer mikrofonu: ")
    try:
        choice_index = int(choice) - 1
        if choice_index < 0 or choice_index >= len(devices):
            raise ValueError("Nieprawidłowy wybór.")
        selected_device = devices[choice_index]
        print(f"Wybrano mikrofon: {selected_device[1]}")

        # Zapisz wybrany mikrofon w ustawieniach
        settings["input_device"] = selected_device[1]
        with open(args.settings, "w") as f:
            yaml.dump(settings, f)
        print(f"Mikrofon zapisany w pliku ustawień: {args.settings}")
    except ValueError as e:
        print(f"Błąd: {e}")
    finally:
        p.terminate()
        exit(0)

# Sprawdzenie, czy input_device jest null
if INPUT_DEVICE_NAME is None:
    print("Brak ustawionego urządzenia wejściowego (mikrofonu).")
    print("Użyj przełącznika '--mic', aby ustawić mikrofon.")
    p.terminate()
    exit(1)

# Wczytanie YAML profilu mapowania
profile_path = os.path.join("profiles", f"{args.profile}.yaml")
with open(profile_path, "r") as f:
    config = yaml.safe_load(f)

# Czas debounce
last_triggered = {}

# Częstotliwości pianina
piano_keys = {
    27.5: "A0", 29.14: "A#0", 30.87: "B0",
    32.7: "C1", 34.65: "C#1", 36.71: "D1", 38.89: "D#1", 41.2: "E1", 43.65: "F1",
    46.25: "F#1", 49.0: "G1", 51.91: "G#1", 55.0: "A1", 58.27: "A#1", 61.74: "B1",
    65.41: "C2", 69.3: "C#2", 73.42: "D2", 77.78: "D#2", 82.41: "E2", 87.31: "F2",
    92.5: "F#2", 98.0: "G2", 103.83: "G#2", 110.0: "A2", 116.54: "A#2", 123.47: "B2",
    130.81: "C3", 138.59: "C#3", 146.83: "D3", 155.56: "D#3", 164.81: "E3", 174.61: "F3",
    185.0: "F#3", 196.0: "G3", 207.65: "G#3", 220.0: "A3", 233.08: "A#3", 246.94: "B3",
    261.63: "C4 (Middle C)", 277.18: "C#4", 293.66: "D4", 311.13: "D#4", 329.63: "E4",
    349.23: "F4", 369.99: "F#4", 392.0: "G4", 415.3: "G#4", 440.0: "A4", 466.16: "A#4",
    493.88: "B4", 523.25: "C5", 554.37: "C#5", 587.33: "D5", 622.25: "D#5", 659.26: "E5",
    698.46: "F5", 739.99: "F#5", 783.99: "G5", 830.61: "G#5", 880.0: "A5", 932.33: "A#5",
    987.77: "B5", 1046.5: "C6", 1108.73: "C#6", 1174.66: "D6", 1244.51: "D#6", 1318.51: "E6",
    1396.91: "F6", 1479.98: "F#6", 1567.98: "G6", 1661.22: "G#6", 1760.0: "A6", 1864.66: "A#6",
    1975.53: "B6", 2093.0: "C7", 2217.46: "C#7", 2349.32: "D7", 2489.02: "D#7", 2637.02: "E7",
    2793.83: "F7", 2959.96: "F#7", 3135.96: "G7", 3322.44: "G#7", 3520.0: "A7", 3729.31: "A#7",
    3951.07: "B7", 4186.01: "C8"
}

# Funkcje sterujące
def press_key(hex_code):
    windll.user32.keybd_event(hex_code, 0, 0, 0)
    
def release_key(hex_code):
    windll.user32.keybd_event(hex_code, 0, KEYEVENTF_UP, 0)
    
def mouse_move(dx, dy): 
    windll.user32.mouse_event(MOUSEEVENTF_MOVE, dx, dy, 0, 0)
    
def mouse_left_click(): 
    windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0); 
    windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    
def mouse_right_click(): 
    windll.user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0); 
    windll.user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
    
def mouse_middle_click(): 
    windll.user32.mouse_event(MOUSEEVENTF_MIDDLEDOWN, 0, 0, 0, 0); 
    windll.user32.mouse_event(MOUSEEVENTF_MIDDLEUP, 0, 0, 0, 0)
    
def modify_sensitivity(delta):
    global SENSITIVITY
    SENSITIVITY = max(1, SENSITIVITY + delta)
    print(f"New sensitivity: {SENSITIVITY}")
    
def reset_sensitivity():
    global SENSITIVITY
    SENSITIVITY = settings.get("sensitivity", 30)
    print("Sensitivity reset")

action_handlers = {
    'mouse_move_up': lambda: mouse_move(0, -SENSITIVITY),
    'mouse_move_down': lambda: mouse_move(0, SENSITIVITY),
    'mouse_move_left': lambda: mouse_move(-SENSITIVITY, 0),
    'mouse_move_right': lambda: mouse_move(SENSITIVITY, 0),
    'mouse_left_click': mouse_left_click,
    'mouse_right_click': mouse_right_click,
    'mouse_middle_click': mouse_middle_click,
    'reset_sensitivity': reset_sensitivity,
}

def handle_action(action_str):
    if action_str.startswith("press_key:"):
        # Wyodrębnij kod HEX i przekonwertuj na liczbę całkowitą
        hex_code = int(action_str.split(":")[1], 16)
        press_key(hex_code)
    elif action_str.startswith("modify_sensitivity:"):
        modify_sensitivity(int(action_str.split(":")[1]))
    else:
        func = action_handlers.get(action_str)
        if func:
            func()
        else:
            print(f"Nieobsługiwana akcja: {action_str}")

def find_nearest_key(freq): return min(piano_keys.keys(), key=lambda x: abs(x - freq))
def frequency_to_note(freq): return piano_keys[find_nearest_key(freq)]

# Inicjalizacja PyAudio
p = pyaudio.PyAudio()
input_index = Nonex
if INPUT_DEVICE_NAME:
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if INPUT_DEVICE_NAME.lower() in info["name"].lower():
            # Sprawdź, czy urządzenie obsługuje RATE
            if info["defaultSampleRate"] == RATE:
                input_index = i
                print(f"Używane urządzenie: {info['name']} (index {i})")
                break
            else:
                print(f"Urządzenie {info['name']} (index {i}) nie obsługuje częstotliwości {RATE} Hz.")
                input_index = i  # Zapisz indeks urządzenia, aby wyświetlić obsługiwane częstotliwości
                break

if input_index is None:
    print(f"Nie znaleziono urządzenia wejściowego pasującego do nazwy: {INPUT_DEVICE_NAME}.")
    p.terminate()
    exit(1)

# Jeśli domyślny RATE nie jest obsługiwany, zapytaj użytkownika o wybór częstotliwości
info = p.get_device_info_by_index(input_index)
if info["defaultSampleRate"] != RATE:
    print(f"Domyślna częstotliwość próbkowania urządzenia '{info['name']}' to {info['defaultSampleRate']} Hz.")
    print("Wybierz jedną z obsługiwanych częstotliwości próbkowania:")
    supported_rates = [8000, 16000, 22050, 32000, 44100, 48000, 96000, 192000]  # Typowe wartości RATE
    valid_rates = [rate for rate in supported_rates if rate <= info["defaultSampleRate"]]
    for idx, rate in enumerate(valid_rates, start=1):
        print(f"{idx}: {rate} Hz")

    choice = input("Wybierz numer częstotliwości: ")
    try:
        choice_index = int(choice) - 1
        if choice_index < 0 or choice_index >= len(valid_rates):
            raise ValueError("Nieprawidłowy wybór.")
        RATE = valid_rates[choice_index]
        print(f"Ustawiono częstotliwość próbkowania na {RATE} Hz.")

        # Zapisz wybraną częstotliwość w ustawieniach
        settings["rate"] = RATE
        with open(args.settings, "w") as f:
            yaml.dump(settings, f)
        print(f"Częstotliwość próbkowania zapisana w pliku ustawień: {args.settings}")
    except ValueError as e:
        print(f"Błąd: {e}")
        p.terminate()
        exit(1)

# Otwórz strumień audio
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=input_index)

frequencies = np.fft.rfftfreq(CHUNK, d=1 / RATE)

def setup_plot():
    fig, ax = plt.subplots(figsize=(10, 5))
    lines = ax.stem(frequencies, np.zeros_like(frequencies))
    peak_text = ax.text(0, 0.8, "", fontsize=12, color="red", ha="left")
    ax.set_xlim(0, 4000)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Częstotliwość (Hz)")
    ax.set_ylabel("Amplituda")
    ax.set_title("Wykrywanie dźwięku")
    return fig, lines, peak_text

def process_audio(lines=None, peak_text=None):
    audio_data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
    volume = np.max(np.abs(audio_data))
    if volume < AMPLITUDE_THRESHOLD:
        if lines: lines[0].set_ydata(np.zeros_like(frequencies))
        if peak_text: peak_text.set_text("")
        return lines if lines else None
    fft_data = np.abs(np.fft.rfft(audio_data)) / CHUNK
    fft_data = fft_data / np.max(fft_data)
    freq = frequencies[np.argmax(fft_data)]
    note = frequency_to_note(freq)
    now = time()
    if (note not in last_triggered or now - last_triggered[note] > TRIGGER_COOLDOWN):
        action = config.get(note)
        if action:
            print(f"Wykryto dźwięk: {note} ({freq:.1f} Hz). Przypisana akcja: {action}")
            handle_action(action)
        else:
            print(f"Wykryto dźwięk: {note} ({freq:.1f} Hz). Brak przypisanej akcji.")
        last_triggered[note] = now
    if lines: lines[0].set_ydata(fft_data)
    if peak_text:
        peak_text.set_text(f"{note} ({freq:.1f} Hz)")
        peak_text.set_position((freq, 0.8))
    return lines if lines else None

if args.plot:
    fig, lines, peak_text = setup_plot()
    ani = animation.FuncAnimation(fig, lambda _: process_audio(lines, peak_text), interval=50, cache_frame_data=False)
    plt.show()
else:
    print("Nasłuchiwanie dźwięków pianina...")
    try:
        while True:
            process_audio()
    except KeyboardInterrupt:
        print("Zatrzymano.")

stream.stop_stream()
stream.close()
p.terminate()
