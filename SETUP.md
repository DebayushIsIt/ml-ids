# Day 1 setup (Windows, about 1 hour)

Days 1 to 4, the API and the dashboard all run on plain Windows. Live capture
(day 5) and Docker (day 8) are easier on Linux; see the last section.

## 1. Install

- **Python 3.11** from python.org (64-bit). In the installer tick
  "Add python.exe to PATH". Check in PowerShell: `py -3.11 --version`.
- **Git for Windows** (git-scm.com).
- **VS Code**, then accept the recommended extensions when you open the
  project (Python, Jupyter, Ruff, Docker).

## 2. Project and virtual environment

Unzip the project to e.g. `C:\Users\<you>\code\ml-ids`. In PowerShell:

```powershell
cd C:\Users\<you>\code\ml-ids
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
pytest -q                        # 10 tests should pass
```

Open the folder in VS Code (`code .`). It should pick up `.venv` automatically;
if not: Ctrl+Shift+P → "Python: Select Interpreter" → `.venv\Scripts\python.exe`.

`make` does not exist on Windows, so use the commands directly:

| Makefile | PowerShell |
|---|---|
| `make test` | `pytest -q` |
| `make lint` | `ruff check src tests` |
| `make data` | `python -m ids.data.clean --raw data\raw\MachineLearningCVE --out data\processed --benign-frac 0.2` |

## 3. Git and GitHub

```powershell
git init -b main
git add .
git commit -m "Day 1: project scaffold and CICIDS2017 cleaning"
# create an empty repo named ml-ids on github.com, then:
git remote add origin https://github.com/<you>/ml-ids.git
git push -u origin main
```

## 4. Dataset

Follow `data/README.md`: download `MachineLearningCSV.zip`, right-click →
Extract All into `data\raw\` so the 8 CSVs sit in `data\raw\MachineLearningCVE\`.
Then run the `make data` command above and open
`notebooks/01_eda_cicids2017.ipynb` (kernel: `.venv`), Run All.

## Day 1 done when

- [ ] `pytest -q` passes
- [ ] `data\processed\cicids2017_clean.parquet` exists
- [ ] the EDA notebook runs top to bottom
- [ ] first commit pushed to GitHub

## Later: live capture and Docker

- **Day 5 live capture.** On Windows, install Npcap (npcap.com) with
  "WinPcap API-compatible mode" ticked so Scapy can sniff. The `cicflowmeter`
  package is Linux-first and may not install or capture on Windows. If it
  fails, either record pcaps with Wireshark and score them in near real time,
  or switch only the sensor to WSL2 (`wsl --install -d Ubuntu-24.04`).
- **Day 8 Docker.** Install Docker Desktop; it enables WSL2 for you. You keep
  working in Windows and PowerShell.
