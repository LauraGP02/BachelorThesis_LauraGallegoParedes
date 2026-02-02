import numpy as np
import scipy.signal as sig
import EntropyHub as EH
from scipy.spatial.distance import pdist

from config import fs, threshold
from preprocessing import compute_zc

# PyRQA imports
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius
from pyrqa.computation import RQAComputation
from pyrqa.metric import EuclideanMetric
from pyrqa.analysis_type import Classic


def zscore(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-10)


def fractal_dimension_boxcount(signal, threshold_img=0.9):
    # Normalize signal to [0, 1]
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-10)
    N = len(signal)

    # Convert 1D signal to a 2D binary image
    img = np.zeros((N, N), dtype=np.uint8)
    for i in range(N):
        y = int(signal[i] * (N - 1))
        img[y, i] = 1

    sizes = np.logspace(1, np.log10(N / 4), num=10, dtype=int)
    counts = []

    for size in sizes:
        size = max(1, size)
        S = (img.shape[0] // size, img.shape[1] // size)
        reduced = img[: S[0] * size, : S[1] * size].reshape(S[0], size, S[1], size)
        reduced = reduced.max(axis=(1, 3))
        counts.append(np.sum(reduced > threshold_img))

    try:
        coeffs = np.polyfit(np.log(1 / sizes), np.log(counts), 1)
        D = coeffs[0]
    except Exception:
        D = np.nan
    return D


def compute_sampen(signal, m=2, r_factor=0.2):
    r = r_factor * np.std(signal)
    Samp, A, B, *_ = EH.SampEn(signal, m=m, tau=1, r=r, Logx=np.exp(1))
    return float(Samp[-1])


def compute_xapen(sig1, sig2, m=2, tau=1, r_factor=0.2):
    pooled_sd = np.sqrt((np.std(sig1) ** 2 + np.std(sig2) ** 2) / 2)
    r = r_factor * pooled_sd
    XAp, Phi = EH.XApEn(sig1, sig2, m=m, tau=tau, r=r, Logx=np.exp(1))
    return float(XAp[-1])


def compute_fuzzy_entropy(signal, m=2, tau=1, r1=0.2, r2=2, Fx="default"):
    r = (r1, r2)
    Fuzz, Ps1, Ps2 = EH.FuzzEn(signal, m=m, tau=tau, Fx=Fx, r=r, Logx=np.exp(1))
    return float(Fuzz[-1])


def cross_correlation(a, b):
    num = np.sum(a * b)
    den = np.sqrt(np.sum(a**2) * np.sum(b**2)) + 1e-10
    return num / den
    

def imaginary_coherency(a, b, fs_local, nperseg=1024):
    f, Pxx = sig.welch(a, fs=fs_local, nperseg=nperseg)
    _, Pyy = sig.welch(b, fs=fs_local, nperseg=nperseg)
    _, Pxy = sig.csd(a, b, fs=fs_local, nperseg=nperseg)
    iCOH = np.imag(Pxy) / (np.sqrt(Pxx * Pyy) + 1e-10)
    return f, iCOH


def magnitude_squared_coherence(x, y, fs_local, nperseg=1024):
    f, mscoh = sig.coherence(x, y, fs=fs_local, nperseg=nperseg)
    return f, mscoh


def radius_optimal(signal, m, delay, RR):
    N = len(signal) - (m - 1) * delay
    if N <= 1:
        return np.nan
    #Each seg reconstructed in phase-space using time-delay embedding
    Y = np.zeros((N, m))
    for i in range(N):
        Y[i, :] = signal[i : i + m * delay : delay]
    #Euclidean distances
    dist = pdist(Y, metric="euclidean")
    epsilon = np.percentile(dist, RR * 100)
    return epsilon


def compute_rqa_2(signal, m=2, delay=3, RR=0.03):
    epsilon = radius_optimal(signal, m=m, delay=delay, RR=RR)
    if np.isnan(epsilon):
        return np.nan, np.nan

    ts = TimeSeries(signal, embedding_dimension=m, time_delay=delay)
    settings = Settings(
        ts,
        analysis_type=Classic,
        neighbourhood=FixedRadius(epsilon),
        similarity_measure=EuclideanMetric,
        theiler_corrector=1,
    )
    computation = RQAComputation.create(settings)
    result = computation.run()
    return result.determinism, result.recurrence_rate

def compute_rqa_1(signal, m=1, delay=1, RR=0.03):
    epsilon = radius_optimal(signal, m=m, delay=delay, RR=RR)
    if np.isnan(epsilon):
        return np.nan, np.nan

    ts = TimeSeries(signal, embedding_dimension=m, time_delay=delay)
    settings = Settings(
        ts,
        analysis_type=Classic,
        neighbourhood=FixedRadius(epsilon),
        similarity_measure=EuclideanMetric,
        theiler_corrector=1,
    )
    computation = RQAComputation.create(settings)
    result = computation.run()
    return result.determinism, result.recurrence_rate


def compute_metrics(rms_center_vmo, rms_center_vl, raw_center_vmo, raw_center_vl, fs_local=fs):
    if (
        rms_center_vmo is None
        or rms_center_vl is None
        or len(rms_center_vmo) == 0
        or len(rms_center_vl) == 0
    ):
        return {k: np.nan for k in [
            "mean_vmo", "median_vmo", "Q1_vmo", "Q3_vmo", "PERC90_vmo",
            "mean_vl", "median_vl", "Q1_vl", "Q3_vl", "PERC90_vl",
            "ratio_vmo/vl", "MAV_vmo", "MAV_vl", "WL_vmo", "WL_vl",
            "ZC_vmo", "ZC_vl",
            "mean_freq_vmo", "median_freq_vmo", "mean_freq_vl", "median_freq_vl",
            "FD_vmo", "FD_vl", "RQA_DET_vmo",
            "RQA_DET_vl", "SampEn_vmo", "SampEn_vl",
            "FuzzyEn_vmo", "FuzzyEn_vl", "XApEn_vmo_vl",
            "Cross-Correlation", "imag_coherency_mean", "Magnitude_squared_coherence_mean",
        ]}

    # ---- Time-domain metrics RMS ----
    mav_vmo = np.mean(np.abs(rms_center_vmo))
    mav_vl  = np.mean(np.abs(rms_center_vl))

    wl_vmo = np.sum(np.abs(np.diff(rms_center_vmo)))
    wl_vl  = np.sum(np.abs(np.diff(rms_center_vl)))

    # Zero-crossings on raw centered segments
    zc_vmo = compute_zc(raw_center_vmo)
    zc_vl = compute_zc(raw_center_vl)
    
    # ---- Frequency-domain metrics RAW ----
    f_vmo, Pxx_vmo = sig.welch(raw_center_vmo, fs=fs_local, detrend=False, nperseg=256, nfft=1024)
    f_vl,  Pxx_vl  = sig.welch(raw_center_vl,  fs=fs_local, detrend=False, nperseg=256, nfft=1024)

    mean_freq_vmo = np.sum(f_vmo * Pxx_vmo) / (np.sum(Pxx_vmo) + 1e-10)
    mean_freq_vl  = np.sum(f_vl  * Pxx_vl)  / (np.sum(Pxx_vl)  + 1e-10)

    cum_power_vmo = np.cumsum(Pxx_vmo)
    cum_power_vl  = np.cumsum(Pxx_vl)

    median_freq_vmo = f_vmo[np.where(cum_power_vmo >= cum_power_vmo[-1] / 2)[0][0]]
    median_freq_vl  = f_vl[np.where(cum_power_vl  >= cum_power_vl[-1]  / 2)[0][0]]

    # ---- Non-linear / complexity metrics ----
    rms_vmo_z=zscore(rms_center_vmo)
    rms_vl_z=zscore(rms_center_vl)
    raw_vmo_z = zscore(raw_center_vmo)
    raw_vl_z  = zscore(raw_center_vl)

    # Fractal dimension
    fd_vmo = fractal_dimension_boxcount(raw_center_vmo)#raw
    fd_vl  = fractal_dimension_boxcount(raw_center_vl)

    # RQA
    det_vmo_1, rec_vmo_1 = compute_rqa_1(raw_vmo_z)#raw normalized
    det_vl_1,  rec_vl_1  = compute_rqa_1(raw_vl_z)
    det_vmo_2, rec_vmo_2 = compute_rqa_2(raw_vmo_z)
    det_vl_2,  rec_vl_2  = compute_rqa_2(raw_vl_z)

    # Entropies
    samp_vmo = compute_sampen(raw_vmo_z)
    samp_vl  = compute_sampen(raw_vl_z)

    fuzz_vmo = compute_fuzzy_entropy(raw_vmo_z)
    fuzz_vl  = compute_fuzzy_entropy(raw_vl_z)

    cross_ap = compute_xapen(raw_vmo_z, raw_vl_z)

    #CC
    CC = cross_correlation(rms_center_vmo, rms_center_vl)#rms

    #iCOH, msCOH
    def band_mean(values_f, values, fmin, fmax):
        band = (values_f >= fmin) & (values_f <= fmax)
        if not np.any(band):
            return np.nan
        return float(np.mean(values[band]))

    f_i, iCOH = imaginary_coherency(raw_vmo_z, raw_vl_z, fs_local)#raw normalized
    iCOH_abs = np.abs(iCOH)
    iCOH_beta  = band_mean(f_i, iCOH_abs, 20, 30)
    iCOH_gamma = band_mean(f_i, iCOH_abs, 30, 70)

    #msCOH
    f_c, mscoh = magnitude_squared_coherence(raw_vmo_z, raw_vl_z, fs_local)
    mscoh_beta  = band_mean(f_c, mscoh, 20, 30)
    mscoh_gamma = band_mean(f_c, mscoh, 30, 70)

    # ---- Summary dictionary ----
    metrics = {
        # Amplitude statistics (RMS)
        "mean_vmo": np.mean(rms_center_vmo),
        "median_vmo": np.median(rms_center_vmo),
        "Q1_vmo": np.percentile(rms_center_vmo, 25),
        "Q3_vmo": np.percentile(rms_center_vmo, 75),
        "PERC90_vmo": np.percentile(rms_center_vmo, 90),

        "mean_vl": np.mean(rms_center_vl),
        "median_vl": np.median(rms_center_vl),
        "Q1_vl": np.percentile(rms_center_vl, 25),
        "Q3_vl": np.percentile(rms_center_vl, 75),
        "PERC90_vl": np.percentile(rms_center_vl, 90),

        "ratio_vmo/vl": np.mean(rms_center_vmo) / (np.mean(rms_center_vl) + 1e-10),

        "MAV_vmo": mav_vmo,
        "MAV_vl": mav_vl,
        "WL_vmo": wl_vmo,
        "WL_vl": wl_vl,

        "ZC_vmo": zc_vmo,
        "ZC_vl": zc_vl,

        # Frequency
        "mean_freq_vmo": mean_freq_vmo,
        "median_freq_vmo": median_freq_vmo,
        "mean_freq_vl": mean_freq_vl,
        "median_freq_vl": median_freq_vl,

        # Non-linear metrics
        "FD_vmo": fd_vmo,
        "FD_vl": fd_vl,
        "RQA_DET_1_vmo": det_vmo_1,
        "RQA_DET_1_vl": det_vl_1,
        "RQA_DET_2_vmo": det_vmo_2,
        "RQA_DET_2_vl": det_vl_2,

        # Entropy
        "SampEn_vmo": samp_vmo,
        "SampEn_vl": samp_vl,
        "FuzzyEn_vmo": fuzz_vmo,
        "FuzzyEn_vl": fuzz_vl,
        "XApEn_vmo_vl": cross_ap,

        # Connectivity
        "Cross-Correlation": CC,
        "iCOH_beta": iCOH_beta,
        "iCOH_gamma": iCOH_gamma,
        "mscoh_beta": mscoh_beta,
        "mscoh_gamma": mscoh_gamma,
    }

    return metrics
