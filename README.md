# Real-Time Acoustic Sonar

A real-time sonar built from a laptop speaker and microphone. It transmits ultrasonic-leaning chirp pulses, matched-filters the echo, and plots the measured distance live — no hardware beyond a sound card.

![Block diagram](assets/block_diagram.png)

## How it works

1. **Chirp generation** — a linear FM chirp (1–15 kHz) is generated as an analytic signal and Hann-windowed to control spectral leakage.
2. **Transmit/receive** — the pulse train is played through the speaker while the microphone records, using `pyaudio` with producer/consumer threads and queues for glitch-free streaming.
3. **Matched filtering** — the received signal is cross-correlated with the conjugate time-reversed chirp (`fftconvolve`), concentrating echo energy into sharp peaks.
4. **Envelope & peak detection** — the Hilbert-transform envelope is searched for the direct-path peak and first echo (`find_peaks`); their sample offset gives round-trip delay.
5. **Range estimation** — delay is converted to distance using the temperature-corrected speed of sound, with outlier rejection and a moving-average smoother, plotted live.

## Repo contents

| File | Description |
|---|---|
| `src/realtime_sonar.py` | Real-time distance measurement with live matplotlib plot |
| `src/sonar_dashboard.py` | Same pipeline rendered as a live Bokeh dashboard (`bokeh serve`) |

## Run it

```bash
pip install -r requirements.txt
python src/realtime_sonar.py            # live matplotlib plot
bokeh serve --show src/sonar_dashboard.py   # browser dashboard
```

Point the speaker/mic at a hard surface within ~20 cm and watch the trace. Adjust `Npulse`, `f0`, `f1`, and `T_air` (air temperature, °C) at the bottom of the script.

## Context

Digital Signal Processing project (ELEC3305, University of Sydney, 2025). All code my own.
