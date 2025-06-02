import os
import glob
import re
import torch
import torchaudio
import numpy as np
from demucs.pretrained import get_model
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─────────────────────────────────────────
# Load fine-tuned model
wrapper = get_model('htdemucs').to(device)
model = wrapper.models[0].to(device)
model.load_state_dict(torch.load('outputs/epoch_30_data_200.th', map_location=device))
model.eval()

# ─────────────────────────────────────────
# SNR 계산 함수
def calculate_snr(clean, estimate):
    noise = clean - estimate
    snr = 10 * torch.log10(torch.sum(clean ** 2) / (torch.sum(noise ** 2) + 1e-10))
    return snr.item()

# ─────────────────────────────────────────
# 숫자 추출용 정렬 함수
def extract_number(filename):
    return int(re.findall(r'\d+', os.path.basename(filename))[0])

# ─────────────────────────────────────────
# 테스트 데이터 경로
noisy_dir = 'test_data/noisy'
clean_dir = 'test_data/clean'
output_dir = 'outputs/denoised_test_v2'
os.makedirs(output_dir, exist_ok=True)

noisy_files = sorted(glob.glob(os.path.join(noisy_dir, '*.wav')), key=extract_number)
clean_files = sorted(glob.glob(os.path.join(clean_dir, '*.wav')), key=extract_number) if os.path.exists(clean_dir) else None

# ✅ 파일 개수 맞는지 확인
if clean_files and len(noisy_files) != len(clean_files):
    raise ValueError(f"File count mismatch: {len(noisy_files)} noisy vs {len(clean_files)} clean")

snr_before_list = []
snr_after_list = []

print("\n✅ File-wise Results:")

for idx, noisy_path in enumerate(tqdm(noisy_files)):
    filename = os.path.basename(noisy_path)
    noisy, sr = torchaudio.load(noisy_path)

    # Convert mono to stereo if needed
    if noisy.shape[0] == 1:
        noisy = noisy.repeat(2, 1)

    noisy_input = noisy.unsqueeze(0).to(device)  # [1, channels, length]

    # Run model
    with torch.no_grad():
        estimate = model(noisy_input).cpu()  # [1, sources, channels, samples]

    # Combine all sources into a single mix: sum across sources
    if estimate.dim() == 4:
        estimate_to_save = estimate[0].sum(dim=0)  # [channels, samples]
    elif estimate.dim() == 3:
        estimate_to_save = estimate[0]
    elif estimate.dim() == 2:
        estimate_to_save = estimate
    else:
        raise ValueError(f"Unexpected tensor shape: {estimate.shape}")

    # Save denoised output
    output_path = os.path.join(output_dir, f"v2_{filename}")
    torchaudio.save(output_path, estimate_to_save, sr)

    # If clean data exists, evaluate SNR
    if clean_files:
        clean, _ = torchaudio.load(clean_files[idx])
        if clean.shape[0] == 1:
            clean = clean.repeat(2, 1)

        min_len = min(clean.shape[-1], estimate_to_save.shape[-1], noisy.shape[-1])
        clean = clean[..., :min_len]
        estimate_trimmed = estimate_to_save[..., :min_len]
        noisy_trimmed = noisy[..., :min_len]

        snr_before = calculate_snr(clean, noisy_trimmed)
        snr_after = calculate_snr(clean, estimate_trimmed)
        snr_before_list.append(snr_before)
        snr_after_list.append(snr_after)

        improvement = snr_after - snr_before
        print(f"{filename}: SNR before = {snr_before:.2f} dB, after = {snr_after:.2f} dB, improvement = {improvement:.2f} dB")
    else:
        print(f"{filename}: Denoised and saved (no clean reference for evaluation)")

# ─────────────────────────────────────────
# Overall summary if clean data exists
if clean_files:
    mean_before = np.mean(snr_before_list)
    mean_after = np.mean(snr_after_list)
    mean_improvement = mean_after - mean_before

    print("\n전체 성능 요약:")
    print(f"평균 SNR before: {mean_before:.2f} dB")
    print(f"평균 SNR after: {mean_after:.2f} dB")
    print(f"평균 improvement: {mean_improvement:.2f} dB")
else:
    print("\n✅ Denoising complete. No clean reference provided, so no SNR evaluation was done.")
