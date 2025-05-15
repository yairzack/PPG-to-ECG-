# utils/remove_baseline_wander.py

from scipy.signal import butter, filtfilt


def remove_baseline_wander(signal, fs, cutoff=0.5, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, signal)