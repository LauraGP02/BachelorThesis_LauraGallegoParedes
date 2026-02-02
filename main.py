import os
import numpy as np
import pandas as pd
from config import fs, data_dir, threshold
from load_data import load_mvic, load_gesture
from preprocessing import (
    bandpass_filter,
    compute_rms,
    rms_step_samples,
    extract_centered_activation,
)
from analysis import compute_metrics


def process_patient(patient_path: str, patient_id: str):
    results = []

    # ---- 1) MVIC ----
    mvic_vmo, mvic_vl = load_mvic(patient_path)

    mvic_vmo_filt = bandpass_filter(mvic_vmo)
    mvic_vl_filt  = bandpass_filter(mvic_vl)

    _, rms_mvic_vmo = compute_rms(mvic_vmo_filt)
    _, rms_mvic_vl  = compute_rms(mvic_vl_filt)

    rms_mvic_vmo = rms_mvic_vmo[~np.isnan(rms_mvic_vmo)]
    rms_mvic_vl  = rms_mvic_vl[~np.isnan(rms_mvic_vl)]

    if len(rms_mvic_vmo) == 0 or len(rms_mvic_vl) == 0:
        print(f"[WARN] MVIC RMS empty for patient {patient_id}. Skipping.")
        return results

    avg_peaks_vmo = np.mean(np.sort(rms_mvic_vmo)[-3:])
    avg_peaks_vl  = np.mean(np.sort(rms_mvic_vl)[-3:])

    # ---- 2) Gestures ----
    for gesture in ["neutral", "adduction"]:
        try:
            emg_vmo, emg_vl = load_gesture(patient_path, gesture)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            continue

        emg_vmo_filt = bandpass_filter(emg_vmo)
        emg_vl_filt  = bandpass_filter(emg_vl)

        rep_dur = 7 * fs
        rest_dur = 60 * fs
        segments = [
            (0, rep_dur),
            (rep_dur + rest_dur, 2 * rep_dur + rest_dur),
            (2 * (rep_dur + rest_dur), 3 * rep_dur + 2 * rest_dur),
        ]

        step = rms_step_samples()

        for rep_index, (start, end) in enumerate(segments, start=1):
            if end > len(emg_vmo_filt) or end > len(emg_vl_filt):
                end = min(len(emg_vmo_filt), len(emg_vl_filt))

            seg_vmo = emg_vmo_filt[start:end]
            seg_vl  = emg_vl_filt[start:end]

            if len(seg_vmo) == 0 or len(seg_vl) == 0:
                print(f"[WARN] Empty segment for patient {patient_id}, gesture {gesture}, rep {rep_index}")
                continue

            _, rms_vmo = compute_rms(seg_vmo)
            rms_vmo = rms_vmo[~np.isnan(rms_vmo)]
            _, rms_vl  = compute_rms(seg_vl)
            rms_vl = rms_vl[~np.isnan(rms_vl)]

            rms_norm_vmo = (rms_vmo / (avg_peaks_vmo + 1e-10)) * 100
            rms_norm_vl  = (rms_vl  / (avg_peaks_vl  + 1e-10)) * 100

            rms_center_vmo, start_vmo, end_vmo = extract_centered_activation(
                rms_norm_vmo, threshold=threshold, window_sec=5, step=step
            )
            rms_center_vl, start_vl, end_vl = extract_centered_activation(
                rms_norm_vl, threshold=threshold, window_sec=5, step=step
            )

            if rms_center_vmo is None or rms_center_vl is None:
                print(f"[INFO] No activation detected for patient {patient_id}, gesture {gesture}, rep {rep_index}")
                continue

            seg_center_start = start + start_vmo * step
            seg_center_end   = start + end_vmo * step

            raw_center_vmo = emg_vmo_filt[seg_center_start:seg_center_end]
            raw_center_vl  = emg_vl_filt[seg_center_start:seg_center_end]

            metrics = compute_metrics(
                rms_center_vmo, rms_center_vl,
                raw_center_vmo, raw_center_vl,
                fs_local=fs,
            )
            metrics.update({
                "patient": patient_id,
                "gesture": gesture,
                "rep": rep_index,
            })
            results.append(metrics)

    return results


def main():
    all_results = []

    for patient in sorted(os.listdir(data_dir)):
        patient_path = os.path.join(data_dir, patient)
        if not os.path.isdir(patient_path):
            continue

        print(f"Processing patient: {patient}")
        patient_results = process_patient(patient_path, patient)
        all_results.extend(patient_results)

    if not all_results:
        print("No results to save.")
        return

    df = pd.DataFrame(all_results)
    first_cols = ["patient", "gesture", "rep"]
    other_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + other_cols]

    df.to_csv(
        "results_emg_metrics_12Enero.csv",
        index=False,
        sep=';',
        decimal=',',
        float_format="%.6f"
    )
    print("Saved results to results_emg_metrics_12Enero.csv")


if __name__ == "__main__":
    main()