# BachelorThesis_LauraGallegoParedes

Python pipeline for sEMG signal preprocessing and feature extraction.

## Overview
This repository contains the scripts used to:
- load mDurance recordings (VMO/VL)
- preprocess raw EMG (band-pass filtering, RMS envelope)
- segment the central activation window
- compute sEMG features (time-domain, frequency-domain, and non-linear metrics)
- export a CSV table for subsequent statistical analysis in SPSS

## Repository structure
- `config.py`: configuration parameters (sampling rate, filter settings, RMS windowing, threshold, data path)
- `load_data.py`: loaders for MVIC and gesture recordings (`neutral`, `adduction`)
- `preprocessing.py`: preprocessing functions (band-pass filter, RMS, activation extraction, ZC)
- `analysis.py`: feature computation (ZC, MNF/MDF, FD, RQA, entropies, coherence metrics)
- `main.py`: batch processing per subject and export of the resulting feature table

## Requirements
Install dependencies:
```bash
Install the required Python packages (NumPy, SciPy, Pandas, EntropyHub, PyRQA).
```

## Data organization

The pipeline expects the input data to be organised on a per-subject basis.
Each subject folder must contain a subdirectory named `mDurance` with the raw
CSV files exported from the mDurance platform.

The expected directory structure is the following:

data/
├── Subject01/
│   └── mDurance/
│       ├── mvic.csv
│       ├── neutral.csv
│       └── adduction.csv
├── Subject02/
│   └── mDurance/
│       ├── mvic.csv
│       ├── neutral.csv
│       └── adduction.csv

Each CSV file is expected to contain two EMG channels corresponding to:
- Vastus Medialis Obliquus (VMO)
- Vastus Lateralis (VL)

The column names are automatically detected based on the presence of the
keywords "vasto_medial" and "vasto_lateral".

## Usage

1. Configure the global parameters in `config.py`, including sampling frequency,
   filter parameters, RMS window length, overlap and activation threshold.

2. Place the input data following the structure described above.

3. Run the main processing script:

```bash
python main.py
```

```md
The pipeline performs the following steps automatically:

- Band-pass filtering of raw sEMG signals (20–450 Hz)
- RMS envelope computation
- MVIC-based normalization
- Automatic detection of the central activation window
- Feature extraction in the time, frequency and non-linear domains
```
The output is a CSV file containing all extracted metrics for each subject,
gesture condition and repetition:

results_emg_metrics_12Enero.csv

## Scope

This repository focuses exclusively on sEMG signal preprocessing and feature
extraction. Statistical analysis and hypothesis testing were performed
separately using dedicated statistical software.
