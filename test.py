import argparse
import yaml
import numpy as np
import sounddevice as sd
import queue
from ctypes import c_uint, windll
from time import time
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

CHUNK = 4096  # Rozmiar bloku audio
RATE = 44100
CHANNELS = 1

KEYEVENTF_UP = 0x0002
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040

parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true", help="Włącz wykres podglądu")
parser.add_argument("--profile", default="default", help="Nazwa profilu mapowania nut (bez .yaml)")
parser.add_argument("--settings", default="settings.yaml", help="Ścieżka do pliku ustawień systemowych")
parser.add_argument("--mic", action="store_true", help="Wylistuj dostępne mikrofony i ustaw jeden z nich")
args = parser.parse_args()

with open(args.settings, "r") as f:
    settings = yaml.safe_load(f)

AMPLITUDE_THRESHOLD = settings.get("amplitude_threshold", 1000)
TRIGGER_COOLDOWN = settings.get("trigger_cooldown", 0.4)
NUM_NOTES = settings.get("num_notes", 3)
SENSITIVITY = settings.get("sensitivity", 30)
INPUT_DEVICE_INDEX = settings.get("input_device_index", None)
RATE = settings.get("rate", RATE)

# Obsługa przełącznika --mic
if args.mic:
    print("Dostępne mikrofony:")
    devices = sd.query_devices()
    input_devices = [(i, d['name']) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
    seen_names = set()
    filtered_devices = []
    for idx, name in input_devices:
        if name not in seen_names:
            seen_names.add(name)
            filtered_devices.append((idx, name))
            print(f"{len(filtered_devices)}: {name}")

    if not filtered_devices:
        print("Nie znaleziono żadnych urządzeń wejściowych.")
        exit(1)

    choice = input("Wybierz numer mikrofonu: ")
    try:
        choice_index = int(choice) - 1
        if choice_index < 0 or choice_index >= len(filtered_devices):
            raise ValueError("Nieprawidłowy wybór.")
        selected_device = filtered_devices[choice_index]
        print(f"Wybrano mikrofon: {selected_device[1]}")
        settings["input_device_index"] = selected_device[0]
        with open(args.settings, "w") as f:
            yaml.dump(settings, f)
        print(f"Mikrofon zapisany w pliku ustawień: {args.settings}")

        # Test mikrofonu przez 3 sekundy
        print("Test mikrofonu przez 3 sekundy...")
        import time
        with sd.InputStream(device=selected_device[0], channels=1, samplerate=RATE, blocksize=CHUNK) as stream:
            for _ in range(int(3 * RATE / CHUNK)):
                data, _ = stream.read(CHUNK)
                volume = np.max(np.abs(data))
                print(f"Volume: {volume:.1f}")
        print("Test zakończony.")
    except ValueError as e:
        print(f"Błąd: {e}")
    exit(0)

profile_path = os.path.join("profiles", f"{args.profile}.yaml")
with open(profile_path, "r") as f:
    config = yaml.safe_load(f)

last_triggered = {}
audio_queue = queue.Queue()


def press_key(hex_code): windll.user32.keybd_event(hex_code, 0, 0, 0)
def release_key(hex_code): windll.user32.keybd_event(hex_code, 0, KEYEVENTF_UP, 0)
def mouse_move(dx, dy): windll.user32.mouse_event(MOUSEEVENTF_MOVE, dx, dy, 0, 0)
def mouse_left_click(): windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0); windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
def mouse_right_click(): windll.user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0); windll.user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
def mouse_middle_click(): windll.user32.mouse_event(MOUSEEVENTF_MIDDLEDOWN, 0, 0, 0, 0); windll.user32.mouse_event(MOUSEEVENTF_MIDDLEUP, 0, 0, 0, 0)

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
        hex_code = int(action_str.split(":")[1], 16)
        press_key(hex_code)
        time.sleep(0.05)
        release_key(hex_code)
    elif action_str.startswith("modify_sensitivity:"):
        modify_sensitivity(int(action_str.split(":")[1]))
    else:
        func = action_handlers.get(action_str)
        if func:
            func()
        else:
            print(f"Nieobsługiwana akcja: {action_str}")

def frequency_to_note(freq):
    if freq <= 0:
        return None
    note_number = 69 + 12 * np.log2(freq / 440.0)
    note_index = int(round(note_number))
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note_name = note_names[note_index % 12]
    octave = note_index // 12 - 1
    return f"{note_name}{octave}"

frequencies = np.fft.rfftfreq(CHUNK, d=1 / RATE)

def audio_callback(indata, frames, time_info, status):
    if status: print(status)
    audio_queue.put(indata.copy())

def setup_plot():
    fig, ax = plt.subplots(figsize=(10, 5))
    markerline, stemlines, baseline = ax.stem(frequencies, np.zeros_like(frequencies))
    peak_text = ax.text(0, 0.8, "", fontsize=12, color="red", ha="left")
    ax.set_xlim(0, 4000)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Częstotliwość (Hz)")
    ax.set_ylabel("Amplituda")
    ax.set_title("Wykrywanie dźwięku")
    return fig, markerline, peak_text

def process_audio(lines=None, peak_text=None):
    if audio_queue.empty(): return
    # print("Przetwarzanie dźwięku...")
    audio_data = audio_queue.get()
    volume = np.max(np.abs(audio_data)) * 1000
    #print(volume)
    if volume < AMPLITUDE_THRESHOLD:
    #     print(f"🔇 Cisza lub zbyt niski poziom dźwięku")
        if lines: lines.set_ydata(np.zeros_like(frequencies))
        if peak_text: peak_text.set_text("")
        return
    mono = audio_data[:, 0]
    fft_data = np.abs(np.fft.rfft(mono)) / CHUNK
    fft_data = fft_data / np.max(fft_data)
    
    top_indices = np.argpartition(fft_data, -NUM_NOTES)[-NUM_NOTES:]  # trzy najmocniejsze częstotliwości
    top_freqs = frequencies[top_indices]
    top_freqs = sorted(top_freqs, key=lambda f: -fft_data[np.where(frequencies == f)[0][0]])

    # Odfiltruj blisko leżące częstotliwości (min 20 Hz odstępu)
    filtered_freqs = []
    for f in top_freqs:
        if all(abs(f - other) > 20 for other in filtered_freqs):
            filtered_freqs.append(f)

    notes = [frequency_to_note(f) for f in filtered_freqs if frequency_to_note(f)]
        
    now = time.time()
    for note in notes:
        # print(f"🔊 Wykryto dźwięk: {note} (amplituda: {volume:.2f})")
        action = config.get(note)
        if action:
            if note not in last_triggered or now - last_triggered[note] > TRIGGER_COOLDOWN:
                print(f"Wykryto dźwięk: {note}. Przypisana akcja: {action}")
                handle_action(action)
                last_triggered[note] = now
        else:
            print(f"Wykryto dźwięk: {note}. Brak przypisanej akcji.")
    if lines: lines.set_ydata(fft_data)
    if peak_text:
        joined = ", ".join(notes)
        peak_text.set_text(f"{joined}")
        peak_text.set_position((top_freqs[0], 0.8))

if __name__ == "__main__":
    # Wyświetl nazwę wybranego mikrofonu
    try:
        input_device_info = sd.query_devices(INPUT_DEVICE_INDEX)
        print(f"Używane urządzenie: {input_device_info['name']} (index {INPUT_DEVICE_INDEX})")
    except Exception:
        print(f"Nie udało się pobrać informacji o urządzeniu o indeksie {INPUT_DEVICE_INDEX}.")
    # Walidacja dostępności mikrofonu
    available_input_indices = {i for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0}
    if INPUT_DEVICE_INDEX not in available_input_indices:
        print(f"Ostrzeżenie: Mikrofon o indeksie {INPUT_DEVICE_INDEX} nie jest już dostępny lub nie obsługuje wejścia audio.")
        print("Użyj opcji --mic, aby wybrać dostępne urządzenie.")
        exit(1)

    print("Nasłuchiwanie dźwięków pianina...")
    with sd.InputStream(callback=audio_callback, channels=CHANNELS, samplerate=RATE, blocksize=CHUNK, device=INPUT_DEVICE_INDEX):
        if args.plot:
            fig, lines, peak_text = setup_plot()
            ani = animation.FuncAnimation(fig, lambda _: process_audio(lines, peak_text), interval=50, cache_frame_data=False)
            plt.show()
        else:
            try:
                while True:
                    process_audio()
            except KeyboardInterrupt:
                print("Zatrzymano.")
