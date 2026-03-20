"""
AI-Harmonized Robotic Band (Team I15)
Pitch Harmonizer & Servo Controller with Strict Noise Gating
"""

import numpy as np
import sounddevice as sd
import queue
import time
from collections import deque

# --- HARDWARE IMPORTS (Raspberry Pi & Mac Fallback) ---
try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685
    from adafruit_motor import servo

    HARDWARE_ENABLED = True
except (ImportError, NotImplementedError):
    print("WARNING: I2C hardware libraries not found. Running in software-only mode.")
    HARDWARE_ENABLED = False

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
FMIN = 27.5
FMAX = 4186.0

# ==========================================
# --- TUNING & CALIBRATION VARIABLES ---
# ==========================================

# Set to True to print volume levels. Play your instrument, note the number,
# then stop playing and note the background noise number. Set VOLUME_THRESHOLD in between.
CALIBRATE_VOLUME = False

# The minimum volume required to trigger pitch detection.
# 0.005 is a quiet room. 0.05 requires loud/close sound.
VOLUME_THRESHOLD = 0.025

# ==========================================

# --- SERVO & HARMONY CONFIGURATION ---
# SERVO_MAP: Which physical PCA9685 channel controls which piano key
SERVO_MAP = {
    60: (0, 45, 0),  # C4 is on Channel 0
    62: (1, 45, 0),  # D4 is on Channel 1
    64: (2, 45, 0),  # E4 is on Channel 2
    65: (3, 45, 0),  # F4 is on Channel 3
    67: (4, 45, 0),  # G4 is on Channel 4
}

# HARMONY_MAP: What the robot plays when it hears you play a specific note
HARMONY_MAP = {
    60: 64,  # If it hears C4 (60), the robot plays E4 (64)
    62: 65,  # If it hears D4 (62), the robot plays F4 (65)
    64: 67,  # If it hears E4 (64), the robot plays G4 (67)
    # Add more harmony rules here!
}


# --- YIN ALGORITHM ---
def yin_pitch(x, fs, threshold=0.15):
    """Pure NumPy YIN algorithm."""
    x = x.astype(np.float64)
    x -= np.mean(x)
    n = len(x)
    min_lag = max(2, int(fs / FMAX))
    max_lag = min(n // 2, int(fs / FMIN) + 1)
    if min_lag >= max_lag: return 0.0

    x_pad = np.zeros(2 * n)
    x_pad[:n] = x
    X = np.fft.rfft(x_pad)
    acf = np.fft.irfft(X * np.conj(X))[:n].real
    energy = float(np.dot(x, x))
    df = np.zeros(max_lag)
    for tau in range(1, max_lag):
        df[tau] = 2.0 * (energy - acf[tau])

    cmnd = np.ones(max_lag)
    running_sum = 0.0
    for tau in range(1, max_lag):
        running_sum += df[tau]
        cmnd[tau] = df[tau] * tau / running_sum if running_sum > 0 else 1.0

    tau_est = -1
    for tau in range(min_lag, max_lag - 1):
        if cmnd[tau] < threshold:
            while tau + 1 < max_lag - 1 and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
            tau_est = tau
            break

    if tau_est == -1:
        tau_est = int(np.argmin(cmnd[min_lag:max_lag])) + min_lag
        if cmnd[tau_est] > 0.4: return 0.0

    if 0 < tau_est < max_lag - 1:
        y0, y1, y2 = cmnd[tau_est - 1], cmnd[tau_est], cmnd[tau_est + 1]
        denom = 2.0 * (2.0 * y1 - y0 - y2)
        if abs(denom) > 1e-10:
            tau_est = tau_est + (y2 - y0) / denom

    if tau_est <= 0: return 0.0
    return fs / tau_est


def freq_to_note(f):
    midi_n = int(round(12 * np.log2(f / 440.0))) + 69
    note_name = NOTES[midi_n % 12]
    return f"{note_name}{(midi_n // 12) - 1}", midi_n


# --- HARDWARE CONTROLLER ---
class ServoController:
    def __init__(self):
        self.servos = {}
        self.active_note = None

        if HARDWARE_ENABLED:
            i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = PCA9685(i2c)
            self.pca.frequency = 50

            for note, (channel, _, rest_angle) in SERVO_MAP.items():
                s = servo.Servo(self.pca.channels[channel])
                s.angle = rest_angle
                self.servos[note] = s

    def play_note(self, midi_n):
        if midi_n == self.active_note: return
        self.release_all()

        if midi_n in SERVO_MAP:
            channel, active_angle, _ = SERVO_MAP[midi_n]
            if HARDWARE_ENABLED:
                self.servos[midi_n].angle = active_angle
            self.active_note = midi_n
            print(f"   [ROBOT] Playing Harmony Note {midi_n} on Servo CH{channel}")

    def release_all(self):
        if self.active_note is not None and self.active_note in SERVO_MAP:
            _, _, rest_angle = SERVO_MAP[self.active_note]
            if HARDWARE_ENABLED:
                self.servos[self.active_note].angle = rest_angle
            self.active_note = None


# --- MAIN RUNNER ---
def main():
    fs = 44100
    frame_size = 4096
    hop_size = 1024

    q = queue.Queue()
    buf = np.zeros(frame_size, dtype=np.float32)

    def callback(indata, frames, time_info, status):
        q.put(indata[:, 0].copy())

    print("\n─── PIANO HARMONIZER & SERVO CONTROLLER ───")
    if CALIBRATE_VOLUME:
        print(">>> CALIBRATION MODE ON: Printing raw volume levels...")

    controller = ServoController()
    note_history = deque(maxlen=3)
    last_printed_note = None

    try:
        with sd.InputStream(samplerate=fs, channels=1, blocksize=hop_size, callback=callback):
            while True:
                chunk = q.get()
                buf[:-hop_size] = buf[hop_size:]
                buf[-hop_size:] = chunk

                rms = float(np.sqrt(np.mean(buf ** 2)))

                if CALIBRATE_VOLUME:
                    print(f"Current Vol: {rms:.4f} | Threshold: {VOLUME_THRESHOLD}")
                    time.sleep(0.1)  # Slow it down so you can read it
                    continue

                if rms < VOLUME_THRESHOLD:
                    note_history.clear()
                    if last_printed_note is not None:
                        controller.release_all()
                        last_printed_note = None
                    continue

                freq = yin_pitch(buf, fs)

                if freq <= FMIN * 1.05 or freq >= FMAX * 0.95:
                    continue

                note_label, heard_midi = freq_to_note(freq)
                note_history.append(heard_midi)

                if len(note_history) == note_history.maxlen and len(set(note_history)) == 1:
                    stable_midi = note_history[0]

                    if stable_midi != last_printed_note:
                        print(f"[MIC] Heard: {note_label} (MIDI: {stable_midi})")

                        # Check if we have a harmony programmed for this note
                        if stable_midi in HARMONY_MAP:
                            harmony_midi = HARMONY_MAP[stable_midi]
                            controller.play_note(harmony_midi)

                        last_printed_note = stable_midi

    except KeyboardInterrupt:
        print("\n\nStopping... resetting servos.")
        controller.release_all()


if __name__ == "__main__":
    main()