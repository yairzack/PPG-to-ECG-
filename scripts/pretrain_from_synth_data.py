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
from utils.remove_baseline_wander import remove_baseline_wander


# def remove_baseline_wander(signal, fs, cutoff=0.5, order=2):
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     b, a = butter(order, normal_cutoff, btype='high', analog=False)
#     return filtfilt(b, a, signal)

def extract_hr_string(filename):
    """Extract heart rate from filename using string methods."""
    # Remove file extension and extract number after 'synthSubHR'
    base_name = os.path.splitext(filename)[0]  # Remove .csv extension
    if base_name.startswith('synthSubHR'):
        hr_str = base_name[10:]  # Remove 'synthSubHR' (10 characters)
        try:
            return int(hr_str)
        except ValueError:
            return None
    return None

# --- Settings ---
window_length = 1024
stride = 256
maxEpochs = 500
batchSize = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Loop over synthetic subjects to pre-train ---
directory = "synth_csv"
for file in os.scandir(directory):
    if not file.is_file() or not file.name.endswith(".csv"):
        continue
    fileHR = extract_hr_string(file.name)
    
    print(f"\n🔵 Processing Synthetic Heart Rate {fileHR}[bpm]...")

    try:
        # --- Load data ---
        datafile = pd.read_csv(file)
        datafile = datafile[['Time [s]', ' II', ' PLETH']]
        
        ecg = datafile[' II'].to_numpy()
        ecg = remove_baseline_wander(ecg, fs=125)  # Assuming fs is the nominal sampling frequency of 125 Hz
        ppg = datafile[' PLETH'].to_numpy()
        ppg = remove_baseline_wander(ppg, fs=125)  # Assuming fs is the nominal sampling frequency of 125 Hz

        if len(ecg) == 0 or len(ppg) == 0 or len(ecg) != len(ppg):
            print(f"Skipping Synthetic Heart Rate {fileHR}[bpm] due to invalid signals.")
            continue

        # --- Normalizing signals to 0-mean and unit std ---
        ecg = (ecg - np.mean(ecg)) / np.std(ecg)
        ppg = (ppg - np.mean(ppg)) / np.std(ppg)

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
            for input_batch, target_batch in tqdm(train_loader, desc=f"Heart Rate {fileHR}[bpm] - Epoch {epoch+1}", unit="batch", leave=False):
                input_batch = input_batch.to(device)
                target_batch = target_batch.to(device)

                output_batch = model(input_batch)
                loss = lossFunction(output_batch, target_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

        # --- Save model ---
        os.makedirs("pretrain_models", exist_ok=True)
        torch.save(model.state_dict(), f"pretrain_models/heart_rate_{fileHR}_pretrain.pth")
        print(f"💾 Model saved: pretrain_models/heart_rate_{fileHR}_pretrain.pth")

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
        eval_df.to_csv(f"pretrain_models/heart_rate_{fileHR}_evaluation.csv", index=False)

        print(f"💾 Evaluation saved: pretrain_models/heart_rate_{fileHR}_evaluation.csv")

    except Exception as e:
        print(f"❌ Error processing Synthetic Heart Rate {fileHR}[bpm]: {e}")
