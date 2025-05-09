# scripts/train_subjects.py

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from models.ecg2ppg_model import ECG2PPGps, create_windows, ECG2PPGps_loss
from utils.eval_metrics import compute_dtw, compute_pearson, compute_rmse
from scipy.signal import butter, filtfilt


def remove_baseline_wander(signal, fs, cutoff=0.5, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, signal)

# --- Settings ---
window_length = 1024
stride = 256
maxEpochs = 500
batchSize = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Loop over subjects ---
# subject_ids = range(1, 54)
subject_ids = [1]  # Example subset for testing

for idx in subject_ids:
    file_idx = f"{idx:02d}"
    file_path = f"bidmc_csv/bidmc_{file_idx}_Signals.csv"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    print(f"\n🔵 Processing Subject {file_idx}...")

    try:
        # --- Load data ---
        datafile = pd.read_csv(file_path)
        datafile = datafile[['Time [s]', ' II', ' PLETH']]
        
        ecg = datafile[' II'].to_numpy()
        ecg = remove_baseline_wander(ecg, fs=125)  # Assuming fs is the nominal sampling frequency of 125 Hz
        ppg = datafile[' PLETH'].to_numpy()
        ppg = remove_baseline_wander(ppg, fs=125)  # Assuming fs is the nominal sampling frequency of 125 Hz

        if len(ecg) == 0 or len(ppg) == 0 or len(ecg) != len(ppg):
            print(f"Skipping subject {file_idx} due to invalid signals.")
            continue

        ecg_windows = create_windows(ecg, window_length, stride)
        ppg_windows = create_windows(ppg, window_length, stride)

        if len(ecg_windows[-1]) != window_length:
            ecg_windows = ecg_windows[:-1]
        if len(ppg_windows[-1]) != window_length:
            ppg_windows = ppg_windows[:-1]

        num_windows = len(ecg_windows)
        split_idx = int(0.8 * num_windows)
        ecg_train, ecg_test = ecg_windows[:split_idx], ecg_windows[split_idx:]
        ppg_train, ppg_test = ppg_windows[:split_idx], ppg_windows[split_idx:]

        # --- DataLoaders ---
        train_dataset = TensorDataset(
            torch.tensor(ppg_train, dtype=torch.float32).unsqueeze(1),
            torch.tensor(ecg_train, dtype=torch.float32).unsqueeze(1)
        )
        test_dataset = TensorDataset(
            torch.tensor(ppg_test, dtype=torch.float32).unsqueeze(1),
            torch.tensor(ecg_test, dtype=torch.float32).unsqueeze(1)
        )

        train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)

        # --- Initialize model ---
        model = ECG2PPGps(in_channels=1, out_channels=1).to(device)
        lossFunction = ECG2PPGps_loss
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.1)

        # --- Training ---
        for epoch in range(maxEpochs):
            model.train()
            for input_batch, target_batch in tqdm(train_loader, desc=f"Subject {file_idx} - Epoch {epoch+1}", unit="batch", leave=False):
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)

                output_batch = model(input_batch)
                loss = lossFunction(output_batch, target_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

        # --- Save model ---
        os.makedirs("subject_models", exist_ok=True)
        torch.save(model.state_dict(), f"subject_models/subject_{file_idx}_model.pth")
        print(f"💾 Model saved: subject_models/subject_{file_idx}_model.pth")

        # --- Evaluation ---
        model.eval()
        dtw_list = []
        pearson_list = []
        rmse_list = []

        reconstructed = []
        true = []

        with torch.no_grad():
            for input_batch, target_batch in test_loader:
                input_batch = input_batch.to(device)
                output_batch = model(input_batch)
                
                output_batch = output_batch.squeeze(1).cpu().numpy()
                target_batch = target_batch.squeeze(1).cpu().numpy()

                reconstructed.append(output_batch)
                true.append(target_batch)

        reconstructed = np.vstack(reconstructed)
        true = np.vstack(true)

        for i in range(reconstructed.shape[0]):
            dtw = compute_dtw(true[i], reconstructed[i])
            pearson = compute_pearson(true[i], reconstructed[i])
            rmse = compute_rmse(true[i], reconstructed[i])

            dtw_list.append(dtw)
            pearson_list.append(pearson)
            rmse_list.append(rmse)

        eval_df = pd.DataFrame({
            'Window': np.arange(len(dtw_list)),
            'DTW_Distance': dtw_list,
            'Pearson_r': pearson_list,
            'RMSE': rmse_list
        })
        eval_df.to_csv(f"subject_models/subject_{file_idx}_evaluation.csv", index=False)

        print(f"💾 Evaluation saved: subject_models/subject_{file_idx}_evaluation.csv")

    except Exception as e:
        print(f"❌ Error processing Subject {file_idx}: {e}")
