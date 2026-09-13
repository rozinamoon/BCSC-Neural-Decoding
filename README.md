# BCSC-Neural-Decoding

## Environment setup

### 1. Clone the repository
```bash
git clone https://github.com/rozinamoon/BCSC-Neural-Decoding.git
cd BCSC-Neural-Decoding
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

### 3. Activate the virtual environment
```bash
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
```

### 4. Install dependencies
```bash
python -m pip install -r requirements.txt
```

## Repository structure
The analysis workflow is divided into three main components:

1. **`preprocessing_matlab/`** — MATLAB/EEGLAB-based preprocessing of raw EEG recordings
2. **`analysis/`** — Python-based quality control, EEG analysis, and visualization of the preprocessed data
