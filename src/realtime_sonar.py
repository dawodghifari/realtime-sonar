import numpy as np, matplotlib.pyplot as plt
import threading,time, queue, pyaudio 
from matplotlib.pyplot import *
import matplotlib.cm as cm
from scipy import signal
from threading import Lock
from scipy.signal import hilbert, find_peaks


def genChirpPulse(Npulse, f0, f1, fs):
    T = Npulse/fs
    t = np.arange(Npulse)/fs
    phi = f0*t + ((f1-f0)/(2*T))*t**2
    return np.exp(1j*2*np.pi*phi)

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
    peaks, _ = signal.find_peaks(seg, distance=52)
    if len(peaks) < 2:
        raise RuntimeError("not enough peaks")
    amps = seg[peaks]
    top2 = peaks[np.argsort(amps)[-2:]]   # two largest peaks
    main, echo = np.sort(top2)[:2]
    return echo - main

def dist2time(dist_cm, T=20):
    v = 331.5*np.sqrt(1 + T/273.15)
    return 2*(dist_cm/100)/v

def time2dist(t_s, T=20):
    v = 331.5*np.sqrt(1 + T/273.15)
    return (v*t_s/2)*100


def play_audio( Q, p, fs , dev=None):
    # play_audio plays audio with sampling rate = fs
    # Q - A queue object from which to play
    # p   - pyAudio object
    # fs  - sampling rate
    # dev - device number
    
    # Example:
    # fs = 44100
    # p = pyaudio.PyAudio() #instantiate PyAudio
    # Q = Queue.queue()
    # Q.put(data)
    # Q.put("EOT") # when function gets EOT it will quit
    # play_audio( Q, p, fs,1 ) # play audio
    # p.terminate() # terminate pyAudio
    
    # open output stream
    ostream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(fs),output=True,output_device_index=dev)
    # play audio
    while (1):
        data = Q.get()
        if data == "EOT" :
            break
        try:
            ostream.write( data.astype(np.float32).tostring() )
        except:
            break
            
def record_audio( queue, p, fs ,dev=None,chunk=2048,lock=None):
    # record_audio records audio with sampling rate = fs
    # queue - output data queue
    # p     - pyAudio object
    # fs    - sampling rate
    # dev   - device number
    # chunk - chunks of samples at a time default 1024
    #
    # Example:
    # fs = 44100
    # Q = Queue.queue()
    # p = pyaudio.PyAudio() #instantiate PyAudio
    # record_audio( Q, p, fs, 1) #
    # p.terminate() # terminate pyAudio
    
    istream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(fs),input=True,input_device_index=dev,frames_per_buffer=chunk)

    # record audio in chunks and append to frames
    frames = [];
    while (1):
        try:  # when the pyaudio object is destroyed, stops
            with lock if lock is not None else 1:
                data_str = istream.read(chunk, exception_on_overflow=False) # read a chunk of data
        except:
            break
        data_flt = np.fromstring( data_str, 'float32' ) # convert string to float
        queue.put( data_flt ) # append to list

def xciever(sig, fs):
    # function takes a signal and a sampling frequency
    # it then plays and records at the same time. The function returns
    # the recorded sound.

    rcv = [];

    # create an input output FIFO queues
    Qin = queue.Queue()
    Qout = queue.Queue()

    #lock for controlling access to shared resources
    lock = Lock()
    
    # create a pyaudio object
    p = pyaudio.PyAudio()

    # initialize a recording thread.
    t_rec = threading.Thread(target = record_audio,   args = (Qin, p, fs ), kwargs={'lock': lock})
    t_play_audio = threading.Thread(target = play_audio,  args = (Qout, p, fs  ))

    # start the recording and playing threads
    t_rec.start()
    t_play_audio.start()

    Qout.put( sig );
    Qout.put( "EOT" );

    # pause for RECORD_SECS seconds
    RECORD_SECS = len(sig)/fs + 0.2
    time.sleep( RECORD_SECS )
    # terminate pyAudio
    with lock:
        p.terminate()
        
    # append to output
    while ( not Qin.empty()) :
        data = Qin.get()
        rcv = np.append( rcv, data )

    return rcv


# ——— New measure‐distance function ———

def measure_distance_one(Npulse, f0, f1, fs, Nseg, temperature=20):
    """
    Send a single chirp, record it, and use findDelay
    to get the sample difference between direct path and first echo.
    Returns one-way distance in cm.
    """
    # 1) build analytic chirp + window + real transmit pulse
    pulse_a = genChirpPulse(Npulse, f0, f1, fs)
    win     = np.hanning(Npulse)
    tx_p    = np.real(pulse_a * win)

    # 2) make one‐segment train and xmit/rcv
    ptrain = genPulseTrain(tx_p, 1, Nseg)      # Nrep=1
    rcv    = xciever(ptrain/1.5, fs)

    # 3) matched‐filter envelope
    X = crossCorr(rcv, pulse_a)
    Xa = np.abs(X)

    # 4) findDelay returns echo_idx – direct_idx (samples)
    delta_samples = findDelay(Xa, Nseg)

    # 5) convert to round‐trip time → one‐way distance
    t_round = delta_samples / fs
    dist_cm = time2dist(t_round, temperature)
    return dist_cm

if __name__ == "__main__":
    Npulse, f0, f1 = 1500, 1000, 15000
    fs, Nseg       = 44100, 12000
    T_air          = 18  # °C

    # prepare interactive plot
    plt.ion()
    fig, ax = plt.subplots()
    valid_line,    = ax.plot([], [], 'bo', label='Valid (≤20 cm)')
    invalid_line,  = ax.plot([], [], 'rx', label='Invalid (>20 cm)')
    smooth_line,   = ax.plot([], [], 'g-', lw=2, label='3-pt MA')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Distance [cm]')
    ax.set_title('Real‐time Sonar Distance')
    ax.legend(loc='upper right')

    xs_valid, ys_valid     = [], []
    xs_invalid, ys_invalid = [], []
    last_valid = None
    start_time = time.time()

    print("Starting real‐time plot with smoothing (Ctrl+C to stop)...")
    try:
        while True:
            now = time.time() - start_time
            d = measure_distance_one(Npulse, f0, f1, fs, Nseg, T_air)

            # categorize measurement
            if d <= 20 or last_valid is None:
                xs_valid.append(now)
                ys_valid.append(d)
                last_valid = d
            else:
                xs_invalid.append(now)
                ys_invalid.append(last_valid)

            # update scatter data
            valid_line.set_data(xs_valid, ys_valid)
            invalid_line.set_data(xs_invalid, ys_invalid)

            # compute 5-point moving average on valid data
            if len(ys_valid) >= 3:
                weights = np.ones(3) / 3
                smoothed = np.convolve(ys_valid, weights, mode='valid')
                x_smooth = xs_valid[2:]  # align with 'valid' output
                smooth_line.set_data(x_smooth, smoothed)

            # auto‐scale x to show new points
            ax.set_xlim(0, max(now, ax.get_xlim()[1]))

            fig.canvas.draw()
            fig.canvas.flush_events()

            time.sleep(0.5)  # one iteration per second

    except KeyboardInterrupt:
        print("\nStopped by user.")
