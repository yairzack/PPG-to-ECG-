# scripts/plot_examples.py

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.ecg2ppg_model import ECG2PPGps, create_windows
from scipy.signal import butter, filtfilt


def remove_baseline_wander(signal, fs, cutoff=0.5, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, signal)

# --- Settings ---
window_length = 1024
stride = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

subject_idx = 1  # change to subject you want
file_idx = f"{subject_idx:02d}"

signal_file = f"bidmc_csv/bidmc_{file_idx}_Signals.csv"
model_file = f"subject_models/subject_{file_idx}_model.pth"

# --- Load data ---
datafile = pd.read_csv(signal_file)
datafile = datafile[['Time [s]', ' II', ' PLETH']]

ecg = datafile[' II'].to_numpy()
ppg = datafile[' PLETH'].to_numpy()

ecg_windows = create_windows(ecg, window_length, stride)
ppg_windows = create_windows(ppg, window_length, stride)

if len(ecg_windows[-1]) != window_length:
    ecg_windows = ecg_windows[:-1]
if len(ppg_windows[-1]) != window_length:
    ppg_windows = ppg_windows[:-1]

split_idx = int(0.8 * len(ecg_windows))
ecg_test = ecg_windows[split_idx:]
ppg_test = ppg_windows[split_idx:]

# --- Remove baseline wander from PPG signal ---
fs = 125  # Nominal sampling frequency (Hz)
ppg_test = np.array([remove_baseline_wander(ppg, fs) for ppg in ppg_test])

ppg_test_tensor = torch.tensor(ppg_test, dtype=torch.float32).unsqueeze(1).to(device)

# --- Load model ---
model = ECG2PPGps(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()

# --- Inference ---
with torch.no_grad():
    reconstructed_test = model(ppg_test_tensor).squeeze(1).cpu().numpy()

# --- Remove baseline wander from ECG signal ---
fs = 125  # Nominal sampling frequency (Hz)
ecg_test = np.array([remove_baseline_wander(ecg, fs) for ecg in ecg_test])
reconstructed_test = np.array([remove_baseline_wander(ecg, fs) for ecg in reconstructed_test])

# --- Normalize signals ---
ecg_test = (ecg_test - np.mean(ecg_test, axis=1, keepdims=True)) / np.std(ecg_test, axis=1, keepdims=True)
reconstructed_test = (reconstructed_test - np.mean(reconstructed_test, axis=1, keepdims=True)) / np.std(reconstructed_test, axis=1, keepdims=True)


# --- Plot random examples ---
num_examples_to_plot = 5
example_indices = np.random.choice(len(ecg_test), num_examples_to_plot, replace=False)

for idx, example_idx in enumerate(example_indices):
    plt.figure(figsize=(14, 4))
    plt.plot(ecg_test[example_idx], label='Ground Truth ECG', linewidth=2)
    plt.plot(reconstructed_test[example_idx], label='Reconstructed ECG', linestyle='--')
    plt.title(f'Subject {file_idx} - Window {example_idx}')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()
