import numpy as np
import threading
import time
import queue
import pyaudio
from scipy import signal
from threading import Lock
from scipy.signal import hilbert, find_peaks

from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Range1d
from bokeh.layouts import column

# ——— Sonar utility functions (identical to before) ———

def genChirpPulse(Npulse, f0, f1, fs):
    T = Npulse / fs
    t = np.arange(Npulse) / fs
    phi = f0 * t + ((f1 - f0) / (2 * T)) * t**2
    return np.exp(1j * 2 * np.pi * phi)

def genPulseTrain(pulse, Nrep, Nseg):
    L = len(pulse)
    if Nseg < L:
        raise ValueError("Nseg must be ≥ pulse length")
    seg = np.zeros(Nseg, dtype=pulse.dtype)
    seg[:L] = pulse
    return np.tile(seg, Nrep)

def crossCorr(rcv, pulse_a):
    return signal.fftconvolve(rcv, np.conj(pulse_a[::-1]), mode='full')

def findDelay(Xrcv_a, Nseg):
    arr = np.ravel(Xrcv_a)
    seg = arr[:Nseg]
    peaks, _ = signal.find_peaks(seg)
    if len(peaks) < 2:
        raise RuntimeError("not enough peaks")
    amps = seg[peaks]
    top2 = peaks[np.argsort(amps)[-2:]]   # indices of two largest peaks
    main, echo = np.sort(top2)[:2]
    return echo - main

def dist2time(dist_cm, T=20):
    v = 331.5 * np.sqrt(1 + T / 273.15)
    return 2 * (dist_cm / 100) / v

def time2dist(t_s, T=20):
    v = 331.5 * np.sqrt(1 + T / 273.15)
    return (v * t_s / 2) * 100

def play_audio(Q, p, fs, dev=None):
    """
    Reads numpy arrays from Q, plays them to the system speaker.
    Stops when it sees the string "EOT".
    """
    ostream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=int(fs),
        output=True,
        output_device_index=dev
    )
    while True:
        data = Q.get()
        if data == "EOT":
            break
        try:
            # convert numpy float32 array to bytes
            ostream.write(data.astype(np.float32).tobytes())
        except:
            break
    ostream.stop_stream()
    ostream.close()

def record_audio(queue_out, p, fs, dev=None, chunk=2048, lock=None):
    """
    Continuously reads chunks from the microphone and pushes float32 numpy arrays into queue_out.
    Exits when PyAudio is terminated externally.
    """
    istream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=int(fs),
        input=True,
        input_device_index=dev,
        frames_per_buffer=chunk
    )
    while True:
        try:
            with lock if lock is not None else threading.Lock():
                data_str = istream.read(chunk, exception_on_overflow=False)
        except Exception:
            break
        # convert raw bytes into numpy float32 array
        data_flt = np.frombuffer(data_str, dtype='float32')
        queue_out.put(data_flt)
    istream.stop_stream()
    istream.close()

def xciever(sig, fs):
    """
    Sends 'sig' out to the speaker and simultaneously records from the mic.
    Returns a single 1D numpy array of float32 samples.
    """
    rcv = np.array([], dtype='float32')
    Qin = queue.Queue()
    Qout = queue.Queue()
    lock = Lock()
    p = pyaudio.PyAudio()

    # Launch the recording thread first
    t_rec = threading.Thread(target=record_audio, args=(Qin, p, fs), kwargs={'lock': lock})
    t_play = threading.Thread(target=play_audio,  args=(Qout, p, fs))

    t_rec.start()
    t_play.start()

    # Push our transmit buffer into Qout, then EOT
    Qout.put(sig)
    Qout.put("EOT")

    # Sleep just long enough for one chirp + small buffer
    RECORD_SECS = len(sig) / fs + 0.2
    time.sleep(RECORD_SECS)

    # Terminate PyAudio so the recording thread ends
    with lock:
        p.terminate()

    # Collect everything that was recorded
    while not Qin.empty():
        data = Qin.get()
        rcv = np.append(rcv, data)

    return rcv

def measure_distance_one(Npulse, f0, f1, fs, Nseg, temperature=20):
    """
    Build a single Hanning‐windowed chirp, transmit it, record the echo,
    matched‐filter it, find the direct and first‐echo peaks, and convert that
    delay into a one‐way distance in cm.
    """
    pulse_a = genChirpPulse(Npulse, f0, f1, fs)
    win    = np.hanning(Npulse)
    tx_p   = np.real(pulse_a * win)

    ptrain = genPulseTrain(tx_p, 1, Nseg)
    rcv    = xciever(ptrain / 1.5, fs)

    X  = crossCorr(rcv, pulse_a)
    Xa = np.abs(X)

    delta_samples = findDelay(Xa, Nseg)
    t_round = delta_samples / fs
    dist_cm = time2dist(t_round, temperature)
    return dist_cm

# ——— Parameters ———
Npulse, f0, f1 = 1500, 1000, 15000
fs, Nseg       = 44100, 12000
T_air          = 18  # °C

# ——— Bokeh data sources & figure setup ———
valid_source   = ColumnDataSource(data=dict(x=[], y=[]))
invalid_source = ColumnDataSource(data=dict(x=[], y=[]))
smooth_source  = ColumnDataSource(data=dict(x=[], y=[]))

p = figure(
    x_axis_label='Time [s]',
    y_axis_label='Distance [cm]',
    title='Real‐time Sonar Distance',
    y_range=Range1d(0, 20),
    x_range=Range1d(0, 10)
)

p.circle('x', 'y', source=valid_source, color='blue', size=6, legend_label='Valid (≤20 cm)')
p.cross('x', 'y', source=invalid_source, color='red', size=8, legend_label='Invalid (>20 cm)')
p.line('x', 'y', source=smooth_source, line_width=2, legend_label='5-pt MA', line_color='green')

p.legend.location = 'top_right'
layout = column(p)

# ——— Shared state between measurement thread and Bokeh callback ———
data_queue = queue.Queue()    # measurement thread will put (timestamp, distance)
xs_valid, ys_valid = [], []
xs_invalid, ys_invalid = [], []
last_valid = None
start_time = time.time()

# ——— Background measurement thread function ———
def measurement_loop():
    global last_valid
    while True:
        try:
            now = time.time() - start_time
            d = measure_distance_one(Npulse, f0, f1, fs, Nseg, T_air)
            # push the (timestamp, distance) into the queue for the Bokeh callback to consume
            data_queue.put((now, d))
        except Exception:
            # If findDelay or PyAudio fails, skip that iteration.
            pass

        time.sleep(0.5)  # measure roughly every 0.5 seconds

# Start the measurement thread (daemon=True so it will exit when the Bokeh server stops)
t_meas = threading.Thread(target=measurement_loop, daemon=True)
t_meas.start()

# ——— Bokeh periodic callback: read any queued measurements and update the plot ———
def update():
    global last_valid

    # Drain the queue of all available measurements
    while not data_queue.empty():
        now, d = data_queue.get()

        # Categorize measurement the same way as before
        if (d <= 20) or (last_valid is None):
            xs_valid.append(now)
            ys_valid.append(d)
            valid_source.stream({'x': [now], 'y': [d]})
            last_valid = d
        else:
            xs_invalid.append(now)
            ys_invalid.append(last_valid)
            invalid_source.stream({'x': [now], 'y': [last_valid]})

    # Recompute the 3‐point moving average whenever there are ≥ 3 valid points
    if len(ys_valid) >= 3:
        weights = np.ones(3) / 3
        smoothed = np.convolve(ys_valid, weights, mode='valid')
        x_smooth = xs_valid[2:]  # align with the 'valid' output
        smooth_source.data = {'x': x_smooth, 'y': smoothed}

    # Auto‐scale the x‐axis if we’ve gone past the right bound
    if xs_valid:
        latest_time = xs_valid[-1]
        if latest_time > p.x_range.end:
            p.x_range.end = latest_time

# Add the periodic callback (runs every 500 ms)
curdoc().add_periodic_callback(update, 500)
curdoc().add_root(layout)
