# scripts/train_multiSubjects.py

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from models.ecg2ppg_model import ECG2PPGps, create_windows, ECG2PPGps_loss
from utils.eval_metrics import compute_dtw, compute_pearson, compute_rmse
from utils.remove_baseline_wander import remove_baseline_wander


# def remove_baseline_wander(signal, fs, cutoff=0.5, order=2):
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     b, a = butter(order, normal_cutoff, btype='high', analog=False)
#     return filtfilt(b, a, signal)

# --- Settings ---
window_length = 1024
stride = 256
maxEpochs = 500
batchSize = 128
trainSetSize = 0.8 # proportion of subjects used for training


# --- Check if GPU is available ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Check if model directory exists ---
if not os.path.exists("trained_models"):
    os.makedirs("trained_models")
    print("Created directory: trained_models")

# --- Check what is the latest model in the directory ---
latest_model = None
file_idx = 1
file_idxs = []
for file in os.listdir("trained_models"):
    if file.startswith("model_") and file.endswith(".pth"):
        try:
            idx = int(file[6:-4])  # Extract the index from the filename
            file_idxs.append(idx)
        except ValueError:
            continue
        latest_model = file
        break
if file_idxs:
    file_idx = max(file_idxs) + 1
    latest_model = f"model_{file_idx:03d}.pth"
    print(f"Latest model found: {latest_model}")
else:
    print("No existing model found. Starting from scratch.")
    latest_model = None

# --- Initialize a new model to avoid overwriting any existing one ---
if latest_model is None:
    print("No existing model found. Initializing a new model.")
    model = ECG2PPGps(in_channels=1, out_channels=1).to(device)
    lossFunction = ECG2PPGps_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.1)
    torch.save(model.state_dict(), f"trained_models/model_{file_idx:03d}.pth")
    print(f"💾 Model initialized and saved: trained_models/model_{file_idx:03d}.pth")
else:
    print("Existing model(s) found. Initializing a new model.")
    model = ECG2PPGps(in_channels=1, out_channels=1).to(device)
    lossFunction = ECG2PPGps_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.1)
    torch.save(model.state_dict(), f"trained_models/model_{file_idx:03d}.pth")
    print(f"💾 Model initialized and saved: trained_models/model_{file_idx:03d}.pth")


# --- Loop over subjects ---
subject_ids = range(1, 54) # Assuming subject IDs are from 1 to 53
subject_ids = range(1, 18) # Example subset for testing
subject_trainIds = subject_ids[:int(len(subject_ids) * trainSetSize)]
subject_testIds = subject_ids[int(len(subject_ids) * trainSetSize):]

for idx in subject_ids:
    subject_idx = f"{idx:02d}"
    file_path = f"bidmc_csv/bidmc_{subject_idx}_Signals.csv"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    print(f"\n🔵 Processing Subject {subject_idx}...")

    try:
        # --- Load data ---
        datafile = pd.read_csv(file_path)
        datafile = datafile[['Time [s]', ' II', ' PLETH']]
        
        ecg = datafile[' II'].to_numpy()
        ecg = remove_baseline_wander(ecg, fs=125)  # Assuming fs is the nominal sampling frequency of 125 Hz
        ppg = datafile[' PLETH'].to_numpy()
        ppg = remove_baseline_wander(ppg, fs=125)  # Assuming fs is the nominal sampling frequency of 125 Hz

        if len(ecg) == 0 or len(ppg) == 0 or len(ecg) != len(ppg):
            print(f"Skipping subject {subject_idx} due to invalid signals.")
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

        # --- Training ---
        for epoch in range(maxEpochs):
            model.train()
            for input_batch, target_batch in tqdm(train_loader, desc=f"Subject {subject_idx} - Epoch {epoch+1}", unit="batch", leave=False):
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)

                output_batch = model(input_batch)
                loss = lossFunction(output_batch, target_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

        # --- Update saved model ---
        torch.save(model.state_dict(), f"trained_models/model_{file_idx:03d}.pth")
        print(f"💾 Model updated: trained_models/model_{file_idx:03d}.pth")

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
        eval_df.to_csv(f"trained_models/model_{file_idx:03d}_subject_{subject_idx}.csv", index=False)

        print(f"💾 Evaluation saved: trained_models/model_{file_idx:03d}_subject_{subject_idx}.csv")

    except Exception as e:
        print(f"❌ Error processing Subject {subject_idx}: {e}")
