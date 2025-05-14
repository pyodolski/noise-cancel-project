import torch
from torch.utils.data import Dataset
import librosa
import os


class DFDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, sample_rate=16000):
        self.noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.wav')])
        self.clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.wav')])
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_files[idx])

        noisy, _ = librosa.load(noisy_path, sr=self.sample_rate)
        clean, _ = librosa.load(clean_path, sr=self.sample_rate)

        noisy_tensor = torch.tensor(noisy).float()
        clean_tensor = torch.tensor(clean).float()

        # 길이 맞추기
        min_len = min(len(noisy_tensor), len(clean_tensor))
        noisy_tensor = noisy_tensor[:min_len]
        clean_tensor = clean_tensor[:min_len]

        return noisy_tensor, clean_tensor
