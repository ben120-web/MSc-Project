import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import os
import io
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# Step 1: Specify the correct path to your client_secrets.json file
CLIENT_SECRETS_PATH = '/Users/benrussell/Frameworks/Google-Drive/client_secret_327364716932-78oacfgib4ilrotdphikdpbkvfnlk76c.apps.googleusercontent.com.json'

if not os.path.isfile(CLIENT_SECRETS_PATH):
    raise FileNotFoundError(f"Client secrets file not found: {CLIENT_SECRETS_PATH}")

# Step 2: Authenticate and create the PyDrive client
gauth = GoogleAuth()
gauth.LoadClientConfigFile(CLIENT_SECRETS_PATH)
gauth.LocalWebserverAuth()  # Creates local webserver and auto handles authentication.
drive = GoogleDrive(gauth)

class SignalDataset(Dataset):
    
    def __init__(self, clean_dir_id, noisy_dir_id, snr_levels, drive):
        self.drive = drive
        self.clean_dir_id = clean_dir_id
        self.noisy_dirs = self.get_snr_folder_ids(noisy_dir_id, snr_levels)
        self.clean_files = self.get_file_ids(clean_dir_id)
        self.noisy_files = [self.get_file_ids(noisy_dir_id) for noisy_dir_id in self.noisy_dirs]

        # Ensure that we only use the minimum number of files present in both directories
        self.min_len = min([len(files) for files in self.noisy_files])

    def __len__(self):
        return len(self.clean_files) * len(self.noisy_dirs)

    def __getitem__(self, idx):
        snr_idx = idx // self.min_len
        file_idx = idx % self.min_len
        
        clean_signal = self.read_h5_file_from_drive(self.clean_files[file_idx])
        noisy_signal = self.read_h5_file_from_drive(self.noisy_files[snr_idx][file_idx])

        return torch.tensor(clean_signal).unsqueeze(0), torch.tensor(noisy_signal).unsqueeze(0)

    def get_snr_folder_ids(self, parent_id, snr_levels):
        folder_ids = {}
        for snr in snr_levels:
            file_list = self.drive.ListFile({'q': f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and title='{snr}' and trashed=false"}).GetList()
            if file_list:
                folder_ids[snr] = file_list[0]['id']
            else:
                raise FileNotFoundError(f"Folder '{snr}' not found in Google Drive under parent ID '{parent_id}'")
        return [folder_ids[snr] for snr in snr_levels]

    def get_file_ids(self, folder_id):
        file_list = self.drive.ListFile({'q': f"'{folder_id}' in parents and mimeType='application/octet-stream' and trashed=false"}).GetList()
        return [file['id'] for file in file_list]

    def read_h5_file_from_drive(self, file_id):
        file = self.drive.CreateFile({'id': file_id})
        file_content = io.BytesIO()
        file.GetContentFile(file_content)
        file_content.seek(0)
        with h5py.File(file_content, 'r') as f:
            data = f['ecgSignal'][:]  # Assuming your dataset has 'ecgSignal' as the key
        return data

def get_folder_id_by_name(folder_name, parent_folder_id=None):
    if parent_folder_id:
        query = f"'{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and title='{folder_name}' and trashed=false"
    else:
        query = f"mimeType='application/vnd.google-apps.folder' and title='{folder_name}' and trashed=false"
    
    file_list = drive.ListFile({'q': query}).GetList()
    if file_list:
        return file_list[0]['id']
    else:
        raise FileNotFoundError(f"Folder '{folder_name}' not found in Google Drive")

def get_dataloader(snr_levels, drive=drive, batch_size=20, shuffle=True, num_workers=2):
    clean_folder_name = 'cleanSignals'
    noisy_folder_name = 'noisySignals'
    
    clean_dir_id = get_folder_id_by_name(clean_folder_name, parent_folder_id=None)
    noisy_dir_id = get_folder_id_by_name(noisy_folder_name, parent_folder_id=None)
    
    dataset = SignalDataset(clean_dir_id, noisy_dir_id, snr_levels, drive)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
