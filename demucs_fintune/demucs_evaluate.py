import os
import glob
import re
import torch
import torchaudio
import numpy as np
from demucs.pretrained import get_model
from tqdm import tqdm

# 디바이스 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─────────────────────────────────────────
# 사전 학습된 모델 불러오기
wrapper = get_model('htdemucs').to(device)
model = wrapper.models[0].to(device)
model.eval()

# ─────────────────────────────────────────
# SNR 계산 함수
def calculate_snr(clean, estimate):
    noise = clean - estimate
    snr = 10 * torch.log10(torch.sum(clean ** 2) / (torch.sum(noise ** 2) + 1e-10))
    return snr.item()

# ─────────────────────────────────────────
# 숫자 순서 정렬용 함수 (예: 1, 2, 10 정렬)
def extract_number(filename):
    numbers = re.findall(r'\d+', os.path.basename(filename))
    return int(numbers[0]) if numbers else 0

# ─────────────────────────────────────────
# 경로 설정
noisy_dir = 'test_data/noisy'
clean_dir = 'test_data/clean'
output_dir = 'outputs/demucs_denoised_test'
os.makedirs(output_dir, exist_ok=True)

# 파일 리스트 불러오기 (숫자 기준 정렬)
noisy_files = sorted(glob.glob(os.path.join(noisy_dir, '*.wav')), key=extract_number)
clean_files = sorted(glob.glob(os.path.join(clean_dir, '*.wav')), key=extract_number) if os.path.exists(clean_dir) else None

# clean, noisy 파일 개수 확인
if clean_files and len(noisy_files) != len(clean_files):
    raise ValueError(f"파일 개수가 다름: noisy {len(noisy_files)}개, clean {len(clean_files)}개")

snr_before_list = []
snr_after_list = []

print("\n파일별 처리 결과:")

for idx, noisy_path in enumerate(tqdm(noisy_files)):
    filename = os.path.basename(noisy_path)
    noisy, sr = torchaudio.load(noisy_path)

    # 모노라면 스테레오로 확장
    if noisy.shape[0] == 1:
        noisy = noisy.repeat(2, 1)

    noisy_input = noisy.unsqueeze(0).to(device)  # [1, 채널, 길이]

    # 모델 추론
    with torch.no_grad():
        estimate = model(noisy_input).cpu()

    # 출력 형태 정리 (sources 합치기)
    if estimate.dim() == 4:
        estimate_to_save = estimate[0].sum(dim=0)
    elif estimate.dim() == 3:
        estimate_to_save = estimate[0]
    elif estimate.dim() == 2:
        estimate_to_save = estimate
    else:
        raise ValueError(f"Unexpected tensor shape: {estimate.shape}")

    # denoised 출력 저장
    output_path = os.path.join(output_dir, f"d_{filename}")
    torchaudio.save(output_path, estimate_to_save, sr)

    # clean 데이터가 있을 때 SNR 평가
    if clean_files:
        clean, _ = torchaudio.load(clean_files[idx])
        if clean.shape[0] == 1:
            clean = clean.repeat(2, 1)

        # 세 신호 길이 맞추기
        min_len = min(clean.shape[-1], estimate_to_save.shape[-1], noisy.shape[-1])
        clean = clean[..., :min_len]
        estimate_trimmed = estimate_to_save[..., :min_len]
        noisy_trimmed = noisy[..., :min_len]

        # SNR 계산
        snr_before = calculate_snr(clean, noisy_trimmed)
        snr_after = calculate_snr(clean, estimate_trimmed)

        snr_before_list.append(snr_before)
        snr_after_list.append(snr_after)

        improvement = snr_after - snr_before
        print(f"{filename}: SNR before = {snr_before:.2f} dB, after = {snr_after:.2f} dB, improvement = {improvement:.2f} dB")
    else:
        print(f"{filename}: Denoised and saved (no clean reference for evaluation)")

# ─────────────────────────────────────────
# 전체 평균 요약
if clean_files:
    mean_before = np.mean(snr_before_list)
    mean_after = np.mean(snr_after_list)
    mean_improvement = mean_after - mean_before

    print("\n전체 성능 요약:")
    print(f"평균 SNR before: {mean_before:.2f} dB")
    print(f"평균 SNR after: {mean_after:.2f} dB")
    print(f"평균 improvement: {mean_improvement:.2f} dB")
else:
    print("\nDenoising complete. No clean reference provided, so no SNR evaluation was done.")
