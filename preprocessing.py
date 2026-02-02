import numpy as np
import pandas as pd
import scipy.signal as sig

from config import fs, lowcut, highcut, N, f0, Q, win_ms, overlap_ms

def bandpass_filter(emg):
    low = lowcut / (fs / 2)
    high = highcut / (fs / 2)
    b, a = sig.butter(N, [low, high], btype='bandpass')
    return sig.filtfilt(b, a, emg)

def compute_rms(emg):
    win_samples = int(fs * win_ms / 1000)
    step = int(win_samples - fs * overlap_ms / 1000)

    t = np.arange(len(emg)) / fs
    emg_series = pd.Series(emg, index=t)

    emg_rms = emg_series.rolling(window=win_samples, center=True).apply(
        lambda x: np.sqrt(np.mean(x ** 2)),
        raw=True,
    )

    rms_downsampled = emg_rms.iloc[::step].reset_index(drop=True)
    time_downsampled = t[::step][:len(rms_downsampled)]
    return np.array(time_downsampled), np.array(rms_downsampled)


def rms_step_samples():
    win_samples = int(fs * win_ms / 1000)
    step = int(win_samples - fs * overlap_ms / 1000)
    return step


def extract_centered_activation(rms_norm, threshold=10, window_sec=5.0, step=None):
    if step is None:
        step = rms_step_samples()

    active_mask = rms_norm > threshold
    active_indexes = np.where(active_mask)[0]

    if len(active_indexes) == 0:
        return None, None, None

    t_start = active_indexes[0]
    t_end = active_indexes[-1]

    # Effective sampling frequency of the RMS envelope
    fs_rms = fs / step

    center = int((t_start + t_end) / 2)
    half_window = int((window_sec / 2) * fs_rms)

    start = max(center - half_window, 0)
    end = min(center + half_window, len(rms_norm))

    return rms_norm[start:end], start, end


def compute_zc(signal, threshold=0.01):
    x = signal - np.mean(signal)
    x1 = x[:-1]
    x2 = x[1:]
    sign_change = x1 * x2 < 0
    amp_ok = (np.abs(x1) > threshold) | (np.abs(x2) > threshold)
    return int(np.sum(sign_change & amp_ok))