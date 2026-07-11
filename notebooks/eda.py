from pathlib import Path

import pandas as pd

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "Placement_Data_Full_Class.csv"

df = pd.read_csv(DATASET_PATH)
print(df.head())
print("\nPlacement rate:")
print(df["status"].value_counts(normalize=True))
print("\nAverage scores by status:")
print(df.groupby("status")[["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]].mean())
