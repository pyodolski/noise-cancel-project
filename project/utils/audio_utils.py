import os
import librosa
import soundfile as sf
import numpy as np

def mix_clean_and_noise(clean_path, noise_path, snr_db=5):
    clean, sr = librosa.load(clean_path, sr=None)
    noise, _ = librosa.load(noise_path, sr=sr)

    # 잡음 길이가 짧으면 반복해서 맞춰줌
    if len(noise) < len(clean):
        repeat = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeat)
    noise = noise[:len(clean)]

    # SNR에 맞게 잡음 스케일 조절
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    k = np.sqrt(clean_power / (10**(snr_db / 10) * noise_power))
    noise = noise * k

    noisy = clean + noise
    return noisy, sr

def create_dataset(clean_dir, noise_dir, out_dir, snr_db=5):
    os.makedirs(out_dir, exist_ok=True)
    clean_files = [f for f in os.listdir(clean_dir) if f.endswith('.wav')]
    noise_files = [f for f in os.listdir(noise_dir) if f.endswith('.wav')]

    for i, clean_file in enumerate(clean_files):
        noise_file = noise_files[i % len(noise_files)]
        noisy, sr = mix_clean_and_noise(
            os.path.join(clean_dir, clean_file),
            os.path.join(noise_dir, noise_file),
            snr_db=snr_db
        )
        sf.write(os.path.join(out_dir, f'noisy_{i}.wav'), noisy, sr)
