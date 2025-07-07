# PPG-to-ECG-
Technion Course project on ECG signal reconstruction form PPG using deep learning
# PPG2ECG-Reconstruction

This repository implements subject-specific deep learning models to reconstruct ECG signals from PPG signals, inspired by the PPG2ECGps approach.

## Project Structure
```text
PPG2ECG-Reconstruction/
│
├── bidmc_clustering.ipynb
│
├── Wnet_depth_width.ipynb
│
├── vitaldb_preproces_test.ipynb
│
├── vitaldb_arythmia_attempt.ipynb
│
├── phsyionet2015_reconstruction.ipynb
│
├── LSTM_pretrain_test.ipynb
│
├── models/ # Model definition + loss + preprocessing
│   └── ecg2ppg_model.py         
│   └── ecg2ppg_LSTM_after.py         
│   └── ecg2ppg_model_6by2.py        
│   └── ecg2ppg_model_depth.py         
│   └── ecg2ppg_model_width.py       
│
├── utils/
│   └── eval_metrics.py          # Evaluation metrics (DTW, Pearson, RMSE)
│
├── scripts/
│   ├── train_subjects.py         # Train one model per subject
│   ├── train_multiSubject.py     # Train one model for multiple subject
│   ├── evaluate_subjects.py      # Evaluate and plot metrics
│   └── plot_examples.py          # Visualize examples
│
│
├── data/                         # (Optional) Folder for raw CSVs
│
├── README.md                     # Project overview
├── requirements.txt              # Python dependencies
└── .gitignore                    # Ignore large or unnecessary files
```


