import numpy as np
import sounddevice as sd
import queue
import time

# notes with no sharps
NATURAL_NOTES = {'C', 'D', 'E', 'F', 'G', 'A', 'B'}

# all 12 notes in music
NOTES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

# estimate the fundamental frequency
def detect_pitch_autocorr(x, fs, fmin=80, fmax=1200):
    x = x.astype(np.float32)
    x = x - np.mean(x)
    x *= np.hanning(len(x)).astype(np.float32)

    c = np.correlate(x, x, mode='full')[len(x)-1:]
    min_lag = int(fs / fmax)
    max_lag = int(fs / fmin)
    if max_lag <= min_lag + 2:
        return 0.0

    region = c[min_lag:max_lag]
    peak_rel = int(np.argmax(region))
    peak = peak_rel + min_lag
    if peak <= 0:
        return 0.0
    
    # turn repeat distance into frequency
    return float(fs / peak)

# change frequency into a MIDI number
def freq_to_midi(f):
    return int(round(12 * np.log2(f / 440.0))) + 69

# turns MIDI number into note letter
def midi_to_letter(n):
    return NOTES[n % 12]

# if note has a sharp, move it to nearest normal note
def snap_to_natural(midi_n):
    name = midi_to_letter(midi_n)
    if name in NATURAL_NOTES:
        return midi_n
    if midi_to_letter(midi_n - 1) in NATURAL_NOTES:
        return midi_n - 1
    if midi_to_letter(midi_n + 1) in NATURAL_NOTES:
        return midi_n + 1
    return midi_n

# move pitch up or down so it stays in same octave
def force_into_range(f, low=150, high=900):
    if f <= 0:
        return 0.0
    while f < low:
        f *= 2.0
    while f > high:
        f /= 2.0
    return f

# check if detected frequency is close enough to a real note
def is_confident_pitch(f, midi_n, confidence_cents=50):
    if f <= 0:
        return False
    
    # expected frequency for the MIDI note
    expected_freq = 440 * (2 ** ((midi_n - 69) / 12.0))
    
    # how many cents off are we?
    cents_off = 1200 * np.log2(f / expected_freq)
    
    return abs(cents_off) <= confidence_cents


def list_and_pick_device():
    devices = sd.query_devices()
    print("\n=== Available Audio Devices ===")
    for i, dev in enumerate(devices):
        print(f"{i}: {dev['name']} (in: {dev['max_input_channels']}, out: {dev['max_output_channels']})")
    print()
    
    try:
        choice = input("Enter device index (default 0): ").strip()
        if choice == "":
            return 0
        return int(choice)
    except (ValueError, IndexError):
        print("Invalid input, using device 0.")
        return 0


def main():
    fs = 44100
    
    # Ask user to pick device or use default
    print("Checking available microphones...")
    device_index = list_and_pick_device()

    # how much sound we check each time
    frame_size = 2048 
    hop_size = 1024 # smaller hop = faster updates

    # pitch limits and octave range
    fmin, fmax = 80, 1200
    fold_low, fold_high = 150, 900

    # ignore super quiet input (tune this based on your mic)
    rms_floor = 0.0010

    # only print when sound is above average loudness by this ratio
    # (allows continuous notes, not just onsets)
    onset_ratio = 1.25          
    onset_min_delta = 0.0005    

    # wait this long before printing another note (debounce on note *changes*)
    debounce_sec = 0.25

    # how many times note must repeat before printing (higher = more stable)
    stable_needed = 2

    # confidence check: how close to a real note (in cents)
    confidence_cents = 60

    q = queue.Queue()
    buf = np.zeros(frame_size, dtype=np.float32)

    rms_avg = 0.0 # average loudness
    last_print_time = 0.0

    # for pitch stability
    last_note = None
    same = 0

    # this runs every time new sound comes in
    def callback(indata, frames, time_info, status):
        q.put(indata[:, 0].copy())

    print(f"\nListening on device {device_index}...")
    print("Playing notes now. Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(
            samplerate=fs,
            channels=1,
            dtype='float32',
            device=device_index,
            blocksize=hop_size,
            callback=callback
        ):
            while True:
                x = q.get()

                # slide old sound out, add new sound in
                buf[:-hop_size] = buf[hop_size:]
                buf[-hop_size:] = x

                # how loud is it?
                rms = float(np.sqrt(np.mean(buf**2)))
                
                # update average loudness with exponential moving average
                if rms_avg == 0:
                    rms_avg = rms
                else:
                    rms_avg = 0.95 * rms_avg + 0.05 * rms

                # skip if below floor
                if rms < rms_floor:
                    continue

                # did it get louder than the rolling average?
                is_onset = rms > rms_avg * onset_ratio

                if not is_onset:
                    continue

                now = time.time()
                
                # only allow a new print every debounce_sec — regardless of note
                if now - last_print_time < debounce_sec:
                    continue

                # find the pitch
                f = detect_pitch_autocorr(buf, fs, fmin=fmin, fmax=fmax)
                if f <= 0:
                    continue

                f = force_into_range(f, low=fold_low, high=fold_high)
                midi_n = freq_to_midi(f)
                midi_n = snap_to_natural(midi_n)
                note = midi_to_letter(midi_n)

                # confidence check: is it really that note?
                if not is_confident_pitch(f, midi_n, confidence_cents=confidence_cents):
                    continue

                # make sure note stays same for a moment
                if note == last_note:
                    same += 1
                else:
                    last_note = note
                    same = 1

                if same < stable_needed:
                    continue

                print(note)
                last_print_time = now

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()