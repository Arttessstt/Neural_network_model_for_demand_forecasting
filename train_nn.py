import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pickle
import os

torch.manual_seed(42)
np.random.seed(42)

WINDOW_SIZE = 7
FORECAST_HORIZON = 1
BATCH_SIZE = 64
EPOCHS = 2000
LEARNING_RATE = 0.001
PATIENCE = 200


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'sales_history.csv')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'model_weights.pt')
SCALER_SAVE_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
except FileNotFoundError:
    print(f"❌ ОШИБКА: Файл не найден: {DATA_PATH}")
    print("Запусти сначала data_generation.py!")
    exit()

df = df.dropna()
df = df.groupby(['date', 'item_id'])['sales_volume'].sum().reset_index()
df = df.sort_values(['item_id', 'date'])

all_X_train, all_y_train = [], []
all_X_val, all_y_val = [], []

sales_scaler = MinMaxScaler()

train_volumes = []
for item_id, group in df.groupby('item_id'):
    group = group.sort_values('date')
    split_idx = int(len(group) * 0.8)
    train_volumes.extend(group['sales_volume'].iloc[:split_idx].values)

sales_scaler.fit(np.array(train_volumes).reshape(-1, 1))

with open(SCALER_SAVE_PATH, 'wb') as f:
    pickle.dump(sales_scaler, f)

def create_sequences(values, window, horizon):
    X, y = [], []
    for i in range(len(values) - window - horizon + 1):
        X.append(values[i : i + window])
        y.append(values[i + window : i + window + horizon])
    return np.array(X), np.array(y)

for item_id, group in df.groupby('item_id'):
    group = group.sort_values('date')
    scaled_values = sales_scaler.transform(group[['sales_volume']]).flatten()
    
    split_idx = int(len(scaled_values) * 0.8)
    train_data = scaled_values[:split_idx]
    val_data = scaled_values[split_idx:]
    
    X_train_i, y_train_i = create_sequences(train_data, WINDOW_SIZE, FORECAST_HORIZON)
    X_val_i, y_val_i = create_sequences(val_data, WINDOW_SIZE, FORECAST_HORIZON)
    
    if len(X_train_i) > 0:
        all_X_train.append(X_train_i)
        all_y_train.append(y_train_i)
    if len(X_val_i) > 0:
        all_X_val.append(X_val_i)
        all_y_val.append(y_val_i)

X_train = np.concatenate(all_X_train, axis=0)
y_train = np.concatenate(all_y_train, axis=0)
X_val = np.concatenate(all_X_val, axis=0)
y_val = np.concatenate(all_y_val, axis=0)

indices = np.random.permutation(len(X_train))
X_train, y_train = X_train[indices], y_train[indices]

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).squeeze(-1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

class DemandModel(nn.Module):
    def __init__(self, input_size):
        super(DemandModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

model = DemandModel(WINDOW_SIZE).to(DEVICE)
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

best_val_mae = float('inf')
patience_counter = 0

print("\n--- Запуск честного обучения ---")
for epoch in range(EPOCHS):
    model.train()
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X_b).squeeze(), y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    val_mae = 0.0
    with torch.no_grad():
        for X_v, y_v in val_loader:
            X_v, y_v = X_v.to(DEVICE), y_v.to(DEVICE)
            pred = model(X_v).squeeze()
            val_mae += nn.L1Loss()(pred, y_v).item() * len(y_v)
        val_mae /= len(X_val)

    scheduler.step(val_mae)

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        patience_counter = 0
    else:
        patience_counter += 1

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1:4d} | Val MAE: {val_mae:.4f} | Best: {best_val_mae:.4f}")

    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

test_preds, test_trues = [], []
with torch.no_grad():
    for X_v, y_v in val_loader:
        X_v = X_v.to(DEVICE)
        test_preds.append(model(X_v).cpu().numpy())
        test_trues.append(y_v.numpy())

test_preds = np.concatenate(test_preds).reshape(-1, 1)
test_trues = np.concatenate(test_trues).reshape(-1, 1)

test_preds_real = sales_scaler.inverse_transform(test_preds)
test_trues_real = sales_scaler.inverse_transform(test_trues)

mae_test = np.mean(np.abs(test_preds_real - test_trues_real))
print(f"\n✅ ИТОГОВЫЙ ЧЕСТНЫЙ MAE: {mae_test:.2f}")

plt.figure(figsize=(10, 5))
plt.plot(test_trues_real[-100:], label='Факт', color='#1f77b4', linewidth=2)
plt.plot(test_preds_real[-100:], label='Прогноз AI', color='#ff7f0e', linestyle='--', linewidth=2)
plt.legend()
plt.title(f"Тестовый прогноз (MAE={mae_test:.2f})")
plt.grid(True, alpha=0.3)
plt.show()