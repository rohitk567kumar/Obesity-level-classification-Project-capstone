# Data folder

The dataset is **not stored in this repository**. Download it with one of these two ways.

**Option 1 (automatic):** from the project folder, run

```bash
python src/download_data.py
```

This saves `data/obesity_data.csv`.

**Option 2 (manual):** download the file from the UCI page, unzip it, and put the CSV in this folder.

https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition

If you use the manual file, run the script with its name:

```bash
python src/train.py --data data/ObesityDataSet_raw_and_data_sinthetic.csv
```

The file has 2,111 rows and 17 columns. The target column is called `NObesity` (or `NObeyesdad` in some versions of the file; the code handles both).
