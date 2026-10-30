"""Download the obesity dataset from the UCI repository and save it as a CSV file.

Run from the project folder:
    python src/download_data.py

The file is saved to data/obesity_data.csv (this folder is ignored by git).
"""

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

# where to save the file
OUTPUT_FILE = Path("data") / "obesity_data.csv"


def main():
    # dataset id 544 = "Estimation of Obesity Levels Based on Eating Habits and Physical Condition"
    print("Downloading the dataset from UCI (needs internet)...")
    dataset = fetch_ucirepo(id=544)

    # join the input columns and the target column into one table
    data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

    # some versions call the target "NObeyesdad"; we use "NObesity" everywhere
    if "NObeyesdad" in data.columns:
        data = data.rename(columns={"NObeyesdad": "NObesity"})

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    data.to_csv(OUTPUT_FILE, index=False)
    print("Saved", data.shape[0], "rows and", data.shape[1], "columns to", OUTPUT_FILE)


if __name__ == "__main__":
    main()
