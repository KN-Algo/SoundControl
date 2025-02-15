import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import scipy.signal
import matplotlib.animation as animation

# 🔹 Mapa częstotliwości klawiszy pianina (A4 = 440 Hz)
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

# Parametry audio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
AMPLITUDE_THRESHOLD = 1000

# Inicjalizacja PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Przygotowanie wykresu
fig, ax = plt.subplots(figsize=(10, 5))
frequencies = np.fft.rfftfreq(CHUNK, d=1 / RATE)
(stem_lines, stem_markers, stem_baseline) = ax.stem(frequencies, np.zeros_like(frequencies))  # Dyskretny wykres
peak_text = ax.text(0, 0.8, "", fontsize=12, color="red", ha="left")

ax.set_xlim(0, 4000)
ax.set_ylim(0, 1)
ax.set_xlabel("Częstotliwość (Hz)")
ax.set_ylabel("Amplituda")
ax.set_title("Rozpoznawanie dźwięków pianina")

# Funkcja do znajdowania najbliższego klawisza
def find_nearest_key(freq):
    return min(piano_keys.keys(), key=lambda x: abs(x - freq))

# Funkcja aktualizacji wykresu
def update(_):
    audio_data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)

    # Sprawdzamy poziom głośności
    volume = np.max(np.abs(audio_data))
    if volume < AMPLITUDE_THRESHOLD:
        stem_lines.set_ydata(np.zeros_like(frequencies))
        peak_text.set_text("")
        return stem_lines,

    # Obliczamy FFT
    fft_data = np.abs(np.fft.rfft(audio_data)) / CHUNK
    fft_data = fft_data / np.max(fft_data)

    # Wykrywanie głównej częstotliwości
    peak_idx = np.argmax(fft_data)
    fundamental_freq = frequencies[peak_idx]

    # Znajdujemy najbliższy klawisz
    nearest_key_freq = find_nearest_key(fundamental_freq)
    key_name = piano_keys[nearest_key_freq]

    # Aktualizacja wykresu
    stem_lines.set_ydata(fft_data)
    peak_text.set_text(f"{key_name} ({nearest_key_freq:.1f} Hz)")
    print(f"{key_name} ({nearest_key_freq:.1f} Hz)")
    peak_text.set_position((nearest_key_freq, 0.8))

    return stem_lines, peak_text

# Animacja Matplotlib
ani = animation.FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)

# Uruchomienie wykresu
plt.show()

# Zatrzymanie PyAudio po zamknięciu okna
stream.stop_stream()
stream.close()
p.terminate()
