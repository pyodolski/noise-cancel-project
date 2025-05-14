import torch.nn as nn

class SimpleDFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=15, padding=7),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=32, num_layers=2, batch_first=True, bidirectional=True)
        self.decoder = nn.Sequential(
            nn.Conv1d(64, 16, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=15, padding=7)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, T)
        enc = self.encoder(x)  # (B, 32, T)
        x_lstm_input = enc.permute(0, 2, 1)  # (B, T, 32)
        lstm_out, _ = self.lstm(x_lstm_input)  # (B, T, 64)
        lstm_out = lstm_out.permute(0, 2, 1)  # (B, 64, T)
        out = self.decoder(lstm_out)  # (B, 1, T)
        return out.squeeze(1)  # (B, T)
