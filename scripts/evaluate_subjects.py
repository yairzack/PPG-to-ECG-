# scripts/evaluate_subjects.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# --- Load all evaluation CSVs ---
eval_files = sorted(glob.glob("subject_models/subject_*_evaluation.csv"))

all_dtw = []
all_pearson = []
all_rmse = []

for file in eval_files:
    df = pd.read_csv(file)
    all_dtw.append(df['DTW_Distance'].values)
    all_pearson.append(df['Pearson_r'].values)
    all_rmse.append(df['RMSE'].values)

all_dtw = np.array(all_dtw, dtype=object)
all_pearson = np.array(all_pearson, dtype=object)
all_rmse = np.array(all_rmse, dtype=object)

mean_dtw = np.array([np.mean(x) for x in all_dtw])
std_dtw = np.array([np.std(x) for x in all_dtw])
mean_pearson = np.array([np.mean(x) for x in all_pearson])
std_pearson = np.array([np.std(x) for x in all_pearson])
mean_rmse = np.array([np.mean(x) for x in all_rmse])
std_rmse = np.array([np.std(x) for x in all_rmse])

subjects = np.arange(1, len(all_dtw) + 1)

# --- Plot ---
fig, axs = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

axs[0].errorbar(subjects, mean_dtw, yerr=std_dtw, fmt='o-', capsize=5, color='blue')
axs[0].set_title('Mean DTW Distance per Subject ± Std')
axs[0].set_ylabel('DTW Distance')
axs[0].grid()

axs[1].errorbar(subjects, mean_pearson, yerr=std_pearson, fmt='o-', capsize=5, color='green')
axs[1].set_title('Mean Pearson Correlation per Subject ± Std')
axs[1].set_ylabel('Pearson r')
axs[1].grid()

axs[2].errorbar(subjects, mean_rmse, yerr=std_rmse, fmt='o-', capsize=5, color='red')
axs[2].set_title('Mean RMSE per Subject ± Std')
axs[2].set_ylabel('RMSE')
axs[2].set_xlabel('Subject Index')
axs[2].grid()

plt.tight_layout()
plt.show()
