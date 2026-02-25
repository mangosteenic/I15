"""
Piano Pitch Detector — Pure NumPy YIN (zero extra dependencies)
Requires only: numpy + sounddevice (already installed)
"""

import numpy as np
import sounddevice as sd
import queue
import time

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
FMIN = 27.5
FMAX = 4186.0


def yin_pitch(x, fs, threshold=0.12):
    """
    YIN algorithm — pure NumPy, no librosa needed.
    Dramatically fewer octave errors than plain autocorrelation.
    Returns frequency in Hz, or 0.0 if no clear pitch detected.
    """
    x = x.astype(np.float64)
    x -= np.mean(x)

    n = len(x)
    min_lag = max(2, int(fs / FMAX))
    max_lag = min(n // 2, int(fs / FMIN) + 1)

    if min_lag >= max_lag:
        return 0.0

    # Step 1: Difference function via FFT (fast)
    # d[tau] = sum((x[t] - x[t+tau])^2) = 2*(energy - autocorr[tau])
    x_pad = np.zeros(2 * n)
    x_pad[:n] = x
    X = np.fft.rfft(x_pad)
    acf = np.fft.irfft(X * np.conj(X))[:n].real
    energy = float(np.dot(x, x))
    df = np.zeros(max_lag)
    for tau in range(1, max_lag):
        df[tau] = 2.0 * (energy - acf[tau])

    # Step 2: Cumulative mean normalization (kills octave errors)
    cmnd = np.ones(max_lag)
    running_sum = 0.0
    for tau in range(1, max_lag):
        running_sum += df[tau]
        cmnd[tau] = df[tau] * tau / running_sum if running_sum > 0 else 1.0

    # Step 3: Find first dip below threshold
    tau_est = -1
    for tau in range(min_lag, max_lag - 1):
        if cmnd[tau] < threshold:
            # Find local minimum around this dip
            while tau + 1 < max_lag - 1 and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
            tau_est = tau
            break

    # If no dip found, take global minimum (less reliable but better than nothing)
    if tau_est == -1:
        tau_est = int(np.argmin(cmnd[min_lag:max_lag])) + min_lag
        if cmnd[tau_est] > 0.35:  # still too noisy — give up
            return 0.0

    # Step 4: Parabolic interpolation for sub-sample accuracy
    if 0 < tau_est < max_lag - 1:
        y0, y1, y2 = cmnd[tau_est - 1], cmnd[tau_est], cmnd[tau_est + 1]
        denom = 2.0 * (2.0 * y1 - y0 - y2)
        if abs(denom) > 1e-10:
            tau_est = tau_est + (y2 - y0) / denom

    if tau_est <= 0:
        return 0.0

    return fs / tau_est


def freq_to_note(f):
    midi_n = int(round(12 * np.log2(f / 440.0))) + 69
    note_name = NOTES[midi_n % 12]
    octave = (midi_n // 12) - 1
    return note_name, midi_n


def list_input_devices():
    print("\nAvailable input devices:")
    for i, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0:
            print(f"  [{i}] {dev['name']}")
    print()


def main():
    fs = 44100

    list_input_devices()
    default_device = sd.default.device[0]

    try:
        raw = input(f"Enter device index (Enter = default [{default_device}]): ").strip()
        device_index = int(raw) if raw else default_device
    except ValueError:
        device_index = default_device

    frame_size = 4096   # ~93ms window
    hop_size   = 512    # controls responsiveness
    rms_floor  = 0.004  # silence gate — raise if you get noise detections

    q = queue.Queue()
    buf = np.zeros(frame_size, dtype=np.float32)

    def callback(indata, frames, time_info, status):
        q.put(indata[:, 0].copy())

    print(f"\n─── PIANO DETECTOR (YIN/numpy) | device {device_index} ───")
    print("Play something — notes appear below. Ctrl+C to stop.\n")

    last_midi = None

    try:
        with sd.InputStream(samplerate=fs, channels=1, device=device_index,
                            blocksize=hop_size, callback=callback):
            while True:
                chunk = q.get()
                buf[:-hop_size] = buf[hop_size:]
                buf[-hop_size:] = chunk

                rms = float(np.sqrt(np.mean(buf ** 2)))
                if rms < rms_floor:
                    if last_midi is not None:
                        print()
                        last_midi = None
                    continue

                freq = yin_pitch(buf, fs)

                if freq <= FMIN * 1.05 or freq >= FMAX * 0.95:
                    continue

                note_label, midi_n = freq_to_note(freq)

                if midi_n != last_midi:
                    print(note_label, end=" ", flush=True)
                    last_midi = midi_n

    except KeyboardInterrupt:
        print("\n\nStopped.")


if __name__ == "__main__":
    main()
