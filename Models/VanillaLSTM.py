import torch
import torch.nn as nn

class SlidingWindowLSTM(nn.Module):
    def __init__(self, input_size=32, hidden_size1=16, hidden_size2=10):
        super(SlidingWindowLSTM, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size1, hidden_size=hidden_size2, batch_first=True)
        
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size2, 5)
        self.fc2 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()  # For binary classification

    def forward(self, x):
        # x: (B, T, F) => e.g., (batch, window_size, num_features)
        out, _ = self.lstm1(x)  # out: (B, T, H1)
        out, _ = self.lstm2(out)  # out: (B, T, H2)

        # Use last time step output (standard for LSTM)
        out = out[:, -1, :]  # (B, H2)

        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        return out