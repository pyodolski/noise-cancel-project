import os
import numpy as np
import soundfile as sf

# ─────────────────────────────────────────
# 두 개의 wav 파일을 섞어 SNR(신호대잡음비)을 맞춰 합성하는 함수
def mix_wavs_clean_length(clean_path, noise_path, output_path, snr_db=0):
    # 깨끗한 음성 파일 읽기
    clean, sr_clean = sf.read(clean_path)
    # 노이즈 음성 파일 읽기
    noise, sr_noise = sf.read(noise_path)

    # 샘플레이트가 다르면 오류 발생
    # 샘플레이트 : 1초당 오디오 신호를 몇 번 측정했는지 나타내는 값
    # 값이 다르면 음원의 해상도와 길이가 서로 맞지 않아서 문제 발생
    if sr_clean != sr_noise:
        raise ValueError(f"샘플레이트 불일치: {clean_path} vs {noise_path}")

    clean_len = len(clean)
    noise_len = len(noise)

    # 노이즈 길이가 깨끗한 음성보다 짧으면 반복해서 길이를 맞춤
    if noise_len < clean_len:
        repeat_times = int(np.ceil(clean_len / noise_len))
        noise = np.tile(noise, repeat_times)[:clean_len]
    else:
        # 길이가 충분하면 앞부분만 사용
        noise = noise[:clean_len]

    # 각 신호의 파워(에너지) 계산 샘플의 값들을 제곱한 뒤 평균
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    # 원하는 SNR에 맞게 노이즈 파워를 조정
    desired_noise_power = clean_power / (10 ** (snr_db / 10))
    scaling_factor = np.sqrt(desired_noise_power / (noise_power + 1e-10))
    noise = noise * scaling_factor

    # 깨끗한 음성과 조정된 노이즈 합성
    noisy = clean + noise
    # 값 범위를 [-1, 1]로 클리핑 (오버플로 방지)
    noisy = np.clip(noisy, -1, 1)

    # 합성된 음성 파일을 저장
    sf.write(output_path, noisy, sr_clean)
    print(f"{os.path.basename(output_path)} 합성 완료")

# ─────────────────────────────────────────
# 폴더 단위로 깨끗한 음성과 노이즈를 짝지어 일괄 처리하는 함수
def batch_mix_clean_noise_ordered(clean_folder, noise_folder, output_folder, snr_db=0):
    # 출력 폴더가 없으면 생성
    os.makedirs(output_folder, exist_ok=True)

    # clean, noise 폴더 내 wav 파일 목록을 정렬해서 불러옴
    clean_files = sorted([f for f in os.listdir(clean_folder) if f.endswith('.wav')])
    noise_files = sorted([f for f in os.listdir(noise_folder) if f.endswith('.wav')])

    # 두 폴더 중 작은 파일 개수만큼 처리
    min_len = min(len(clean_files), len(noise_files))
    print(f"총 {min_len}쌍 처리 시작 (clean {len(clean_files)}개, noise {len(noise_files)}개)")

    # 각 파일쌍에 대해 순차적으로 합성 수행
    for idx in range(min_len):
        clean_file = clean_files[idx]
        noise_file = noise_files[idx]

        clean_path = os.path.join(clean_folder, clean_file)
        noise_path = os.path.join(noise_folder, noise_file)
        # 출력 파일 이름은 001.wav, 002.wav처럼 생성
        output_filename = f"{idx + 1:05d}.wav"
        output_path = os.path.join(output_folder, output_filename)

        try:
            # 개별 합성 함수 호출
            mix_wavs_clean_length(clean_path, noise_path, output_path, snr_db)
        except Exception as e:
            # 에러 발생 시 메시지 출력
            print(f"{output_filename} 처리 실패: {e}")

# ─────────────────────────────────────────
# 사용 예시
batch_mix_clean_noise_ordered('data2/clean', 'data2/noise', 'data2/noisy', snr_db=5)
