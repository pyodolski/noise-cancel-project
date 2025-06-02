import os
import glob
import re
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from demucs.pretrained import get_model
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─────────────────────────────────────────
# Custom Dataset
class AudioDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, sample_rate=44100):
        def extract_number(filename):
            return int(re.findall(r'\d+', os.path.basename(filename))[0])

        self.noisy_files = sorted(glob.glob(os.path.join(noisy_dir, '*.wav')), key=extract_number)
        self.clean_files = sorted(glob.glob(os.path.join(clean_dir, '*.wav')), key=extract_number)
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_path = self.noisy_files[idx]
        clean_path = self.clean_files[idx]

        print(f"[DEBUG] Loading Noisy: {os.path.basename(noisy_path)}, Clean: {os.path.basename(clean_path)}")

        noisy, _ = torchaudio.load(noisy_path)
        clean, _ = torchaudio.load(clean_path)

        # Ensure same length
        min_len = min(noisy.shape[-1], clean.shape[-1])
        noisy = noisy[..., :min_len]
        clean = clean[..., :min_len]

        # Convert mono to stereo if needed
        if noisy.shape[0] == 1:
            noisy = noisy.repeat(2, 1)
        if clean.shape[0] == 1:
            clean = clean.repeat(2, 1)

        return noisy, clean

# ─────────────────────────────────────────
# Load pre-trained BagOfModels and pick first model
wrapper = get_model('htdemucs').to(device)
wrapper.train()

model = wrapper.models[0].to(device)
model.train()

# ─────────────────────────────────────────
# Prepare data
dataset = AudioDataset('data/noisy', 'data/clean')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# ─────────────────────────────────────────
# Optimizer + Loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.L1Loss()

# ─────────────────────────────────────────
# Training loop
epochs = 10
for epoch in range(epochs):
    epoch_loss = 0
    for noisy, clean in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        noisy, clean = noisy.to(device), clean.to(device)

        optimizer.zero_grad()
        estimates = model(noisy)  # [1, 4, 2, N]
        estimates_mixed = estimates.sum(dim=1)  # [1, 2, N] → sum over sources

        loss = loss_fn(estimates_mixed, clean)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader):.4f}")

# ─────────────────────────────────────────
# Save fine-tuned weights
os.makedirs('outputs', exist_ok=True)
torch.save(model.state_dict(), 'outputs/epoch_10_data_100.th')
print("✅ Fine-tuned model saved to outputs/finetuned_model_v2.th")
