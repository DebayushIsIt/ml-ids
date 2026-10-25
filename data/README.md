# Data

Raw and processed data are gitignored. To reproduce:

## CICIDS2017 (MachineLearningCVE)

1. Go to https://www.unb.ca/cic/datasets/ids-2017.html and click the download
   link at the bottom. The site asks for name and email, then shows a file list.
2. Download `MachineLearningCSV.zip` (about 225 MB, 8 CSVs, ~2.8M flows).
3. Unzip so the CSVs sit in `data/raw/MachineLearningCVE/`:

   ```bash
   unzip MachineLearningCSV.zip -d data/raw/
   ls data/raw/MachineLearningCVE/
   # Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
   # Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
   # Friday-WorkingHours-Morning.pcap_ISCX.csv
   # Monday-WorkingHours.pcap_ISCX.csv
   # Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
   # Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
   # Tuesday-WorkingHours.pcap_ISCX.csv
   # Wednesday-workingHours.pcap_ISCX.csv
   ```

   If the zip unpacks to a different folder name, pass it with
   `python -m ids.data.clean --raw <folder>`.

If the CIC site is down, the same CSVs are mirrored on Kaggle (search
"CICIDS2017 MachineLearningCVE"). Check that you get the 8 files above.

## Cleaning

```bash
python -m ids.data.clean --raw data/raw/MachineLearningCVE --out data/processed --benign-frac 0.2
```

Writes `data/processed/cicids2017_clean.parquet` and `cleaning_summary.json`.
Use `--benign-frac 1.0` to keep all benign rows.
