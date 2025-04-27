# PPG-to-ECG-
Technion Course project on ECG signal reconstruction form PPG using deep learning
# PPG2ECG-Reconstruction

This repository implements subject-specific deep learning models to reconstruct ECG signals from PPG signals, inspired by the PPG2ECGps approach.

## Project Structure
PPG2ECG-Reconstruction/
├── models/
          └── ecg2ppg_model.py # Model definition + loss + preprocessing 
├── utils/
          └── eval_metrics.py # Evaluation metrics (DTW, Pearson, RMSE)
├── scripts/
            ├── train_subjects.py # Train one model per subject 
            ├── evaluate_subjects.py # Evaluate all subjects and plot metrics 
            ├── plot_examples.py # Visualize examples of reconstructed ECG 
├── subject_models/ # Saved models and evaluations (created after training) 
├── data/ # (optional) Where raw data would be stored (NOT pushed) 
├── requirements.txt # Python dependencies 
├── .gitignore # Ignore unnecessary files └── README.md # Project overview

