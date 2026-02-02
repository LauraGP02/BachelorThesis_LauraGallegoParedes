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
pip install -r requirements.txt
