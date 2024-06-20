import h5py
from torch.utils.data import Dataset, DataLoader
import os

class SignalDataset(Dataset):
    
    def __init__(self, clean_dir, noisy_dir, snr_levels):
        self.clean_dir = clean_dir
        self.noisy_dirs = [os.path.join(noisy_dir, snr) for snr in snr_levels]
        self.clean_files = sorted(os.listdir(clean_dir))
        self.noisy_files = [sorted(os.listdir(noisy_dir)) for noisy_dir in self.noisy_dirs]

        # Ensure that we only use the minimum number of files present in both directories
        self.min_len = min([len(files) for files in self.noisy_files])

    def __len__(self):
        return len(self.clean_files) * len(self.noisy_dirs)

    def __getitem__(self, idx):
        snr_idx = idx // self.min_len
        file_idx = idx % self.min_len
        
        clean_path = os.path.join(self.clean_dir, self.clean_files[file_idx])
        noisy_path = os.path.join(self.noisy_dirs[snr_idx], self.noisy_files[snr_idx][file_idx])

        clean_signal = self.read_h5_file(clean_path)
        noisy_signal = self.read_h5_file(noisy_path)

        return clean_signal, noisy_signal

    @staticmethod
    def read_h5_file(file_path):
        with h5py.File(file_path, 'r') as f:
            data = f['data'][:]
        return data

def get_dataloader(clean_dir, noisy_dir, snr_levels, batch_size=20, shuffle=True, num_workers=2):
    dataset = SignalDataset(clean_dir, noisy_dir, snr_levels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
