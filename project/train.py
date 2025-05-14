import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from models.dfnet_model import SimpleDFNet
from datasets.df_dataset import DFDataset

print("🚀 훈련 시작 준비 중...")
# 하이퍼파라미터 설정
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 0.001
SAMPLE_RATE = 16000

# collate_fn: 길이 안 맞는 오디오들을 자른 뒤 batch로 묶기
def collate_fn(batch):
    min_len = min(min(len(noisy), len(clean)) for noisy, clean in batch)
    noisy_batch = torch.stack([noisy[:min_len] for noisy, _ in batch])
    clean_batch = torch.stack([clean[:min_len] for _, clean in batch])
    return noisy_batch, clean_batch

# 디바이스 설정 (CUDA 우선)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 및 DataLoader 정의
dataset = DFDataset(
    noisy_dir='data/mixtures_train',
    clean_dir='data/clean_train',
    sample_rate=SAMPLE_RATE
)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn  # 🔥 길이 문제 해결
)

# 모델 정의 및 학습 세팅
model = SimpleDFNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 학습 루프


for epoch in range(EPOCHS):
    print(f"📦 Epoch {epoch + 1} 시작")
    model.train()
    total_loss = 0.0

    for noisy_batch, clean_batch in dataloader:
        print("👉 배치 로드됨")
        noisy_batch = noisy_batch.to(device)
        clean_batch = clean_batch.to(device)

        # 순전파
        output = model(noisy_batch)

        # 손실 계산 및 역전파
        loss = criterion(output, clean_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

# 학습된 모델 저장
torch.save(model.state_dict(), "trained_dfnet.pt")
print("✅ 모델 저장 완료: trained_dfnet.pt")
