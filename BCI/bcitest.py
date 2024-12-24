import torch
import torch.nn as nn
import pandas as pd
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CNN_BiLSTM_MultiheadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_heads=8):
        super(CNN_BiLSTM_MultiheadAttention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.lstm = nn.LSTM(256, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.7)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=num_heads, dropout=0.7)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.transpose(1, 2)
        out, (hn, cn) = self.lstm(x)
        out = out.transpose(0, 1)
        attn_out, _ = self.attn(out, out, out)
        attn_out = attn_out.transpose(0, 1)
        out = self.fc(attn_out[:, -1, :])
        return out

model = CNN_BiLSTM_MultiheadAttention(input_size=14, hidden_size=64, num_layers=3, output_size=1).to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()

def test_model(input_data):
    start_time = time.time()
    input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
    input_tensor = input_tensor.unsqueeze(0)  
    with torch.no_grad():
        output = model(input_tensor)
    inference_time = time.time() - start_time
    return torch.sigmoid(output).item(), inference_time


input_data = pd.read_csv('input.csv').values
output, inference_time = test_model(input_data)
print(f"Output: {output:.4f}, Inference Time: {inference_time:.4f} seconds")