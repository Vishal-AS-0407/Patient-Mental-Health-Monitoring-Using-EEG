import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.cuda.amp as amp
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingLR

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
        self._initialize_weights()

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

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

epochs_data = np.load('epochs.npy')
labels = pd.read_csv('TrainLabels.csv')
y = labels.values.flatten()

X = epochs_data
X = np.transpose(X, (0, 2, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

input_size = 14
hidden_size = 64
num_layers = 3
output_size = 1
batch_size = 32
num_epochs = 50
accumulation_steps = 4

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = CNN_BiLSTM_MultiheadAttention(input_size, hidden_size, num_layers, output_size).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.BCEWithLogitsLoss()

scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
scaler = amp.GradScaler()

def train_model():
    model.train()
    epoch_start_time = time.time()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        optimizer.zero_grad()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
            scaler.scale(loss).backward()
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct_preds += (predicted.squeeze() == targets).sum().item()
            total_preds += targets.size(0)
        train_accuracy = correct_preds / total_preds
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.4f}")
        scheduler.step()
        torch.cuda.empty_cache()
    epoch_end_time = time.time()
    print(f"Training time for {num_epochs} epochs: {epoch_end_time - epoch_start_time:.2f} seconds")

def evaluate_model():
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predicted = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(predicted)
            all_labels.extend(targets.cpu().numpy())
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    accuracy = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    f1 = f1_score(all_labels, (all_preds > 0.5).astype(int))
    auc = roc_auc_score(all_labels, all_preds)
    precision = precision_score(all_labels, (all_preds > 0.5).astype(int))
    recall = recall_score(all_labels, (all_preds > 0.5).astype(int))
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def main():
    train_model()
    evaluate_model()
    save_model(model)

if __name__ == "__main__":
    main()
