import os
import pandas as pd


def _load_mdurance_csv(csv_path: str):
    df = pd.read_csv(csv_path, delimiter=';', decimal=',')
    cols = df.columns

    vmo_col = next(c for c in cols if "vasto_medial" in c.lower())
    vl_col  = next(c for c in cols if "vasto_lateral" in c.lower())

    vmo = df[vmo_col]
    vl  = df[vl_col]
    return vmo, vl


def load_mvic(patient_path: str):
    mvic_path = os.path.join(patient_path, "mDurance", "mvic.csv")
    if not os.path.isfile(mvic_path):
        raise FileNotFoundError(f"MVIC file not found: {mvic_path}")
    return _load_mdurance_csv(mvic_path)


def load_gesture(patient_path: str, gesture: str):
    csv_path = os.path.join(patient_path, "mDurance", f"{gesture}.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Gesture file not found: {csv_path}")
    return _load_mdurance_csv(csv_path)
