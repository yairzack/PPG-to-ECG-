# PPG-to-ECG-
Technion Course project on ECG signal reconstruction form PPG using deep learning
# PPG2ECG-Reconstruction

This repository implements subject-specific deep learning models to reconstruct ECG signals from PPG signals, inspired by the PPG2ECGps approach.

## Project Structure
```text
PPG2ECG-Reconstruction/
│
├── models/
│   └── ecg2ppg_model.py         # Model definition + loss + preprocessing
│
├── utils/
│   └── eval_metrics.py          # Evaluation metrics (DTW, Pearson, RMSE)
│
├── scripts/
│   ├── train_subjects.py         # Train one model per subject
│   ├── evaluate_subjects.py      # Evaluate and plot metrics
│   └── plot_examples.py          # Visualize examples
│
├── subject_models/               # Saved models and evaluations (created after training)
│
├── data/                         # (Optional) Folder for raw CSVs
│
├── README.md                     # Project overview
├── requirements.txt              # Python dependencies
└── .gitignore                    # Ignore large or unnecessary files
```
## How to Use

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
2. Train model per subject and output wieghts to target folder
   ```bash
   python scripts/train_subjects.py
3. Evaluate models and plot performance
   ```bash
   python scripts/evaluate_subjects.py

