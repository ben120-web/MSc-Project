# data_loading.py

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import h5py

def normalise_signal(signal):
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    normalised_signal = (signal - signal_min) / (signal_max - signal_min)
    return normalised_signal

def load_h5_file(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            data = f['ecgSignal'][:]
            data = normalise_signal(data)
            print(f"Loaded {file_path}, shape: {data.shape}")
            return data
    except OSError as e:
        print("Unable to open file" + file_path)
        return None

class ECGDataset(Dataset):
    def __init__(self, clean_signals, noisy_signals, segment_length=1500):
        self.clean_signals = clean_signals
        self.noisy_signals = noisy_signals
        self.signal_numbers = list(clean_signals.keys())
        self.segment_length = segment_length

    def __len__(self):
        total_segments = 0
        for signal_number in self.signal_numbers:
            signal_length = self.clean_signals[signal_number].size
            segments_per_signal = signal_length // self.segment_length
            total_segments += segments_per_signal * len(self.noisy_signals)
        return total_segments

    def __getitem__(self, idx):
        segments_per_signal = 10  # Assuming 15000 samples per signal and 1500 samples per segment
        signal_idx = idx // (segments_per_signal * len(self.noisy_signals))
        if signal_idx >= len(self.signal_numbers):
            raise IndexError(f"Signal index {signal_idx} out of range for available signals")
        segment_in_signal_idx = idx % (segments_per_signal * len(self.noisy_signals))
        snr_idx = segment_in_signal_idx // segments_per_signal
        segment_idx = segment_in_signal_idx % segments_per_signal

        signal_number = self.signal_numbers[signal_idx]
        snr = list(self.noisy_signals.keys())[snr_idx]

        start_idx = segment_idx * self.segment_length
        end_idx = start_idx + self.segment_length

        if signal_number in self.noisy_signals[snr]:
            clean_signal_data = self.clean_signals[signal_number][0]
            clean_signal_segment = clean_signal_data[start_idx:end_idx]
            noisy_signal_segments = []
            for noisy_signal_copy in self.noisy_signals[snr][signal_number]:
                noisy_signal_segment = noisy_signal_copy[0][start_idx:end_idx]
                noisy_signal_segments.append(noisy_signal_segment)

            clean_signal_tensor = torch.tensor(clean_signal_segment).unsqueeze(0)
            noisy_signal_tensor = torch.tensor(noisy_signal_segments)

            clean_signal_tensor = clean_signal_tensor.unsqueeze(0)
            noisy_signal_tensor = noisy_signal_tensor.unsqueeze(1)

            return clean_signal_tensor, noisy_signal_tensor

        return torch.tensor([]), torch.tensor([])

def load_data(clean_signals_path, noisy_signals_path, segment_length=1500, batch_size=1, num_workers=2):
    clean_signals = {}
    for file in glob.glob(os.path.join(clean_signals_path, '*.h5')):
        signal_number = os.path.basename(file).split('_')[1]
        clean_signals[signal_number] = load_h5_file(file)

    noisy_signals = {snr: {} for snr in ['SNR0', 'SNR12', 'SNR18', 'SNR24']}
    for snr in noisy_signals.keys():
        snr_path = os.path.join(noisy_signals_path, snr)
        for file in glob.glob(os.path.join(snr_path, '*.h5')):
            signal_number = os.path.basename(file).split('-')[0].split('_')[1]
            if signal_number not in noisy_signals[snr]:
                noisy_signals[snr][signal_number] = []
            noisy_signals[snr][signal_number].append(load_h5_file(file))

    dataset = ECGDataset(clean_signals, noisy_signals, segment_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return dataloader
