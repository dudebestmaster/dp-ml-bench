from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from opacus import PrivacyEngine

# -----------------------------
# 1. Load Adult dataset
# -----------------------------
adult = fetch_ucirepo(id=2)

X = adult.data.features
y = adult.data.targets.iloc[:, 0]


y = y.str.strip().str.replace('.', '', regex=False)
y = (y == ">50K").astype(int).values


X = X.replace('?', np.nan)
mask = X.notna().all(axis=1)
X = X[mask]
y = y[mask]


X = pd.get_dummies(X)

# -----------------------------
# 2. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# -----------------------------
# 3. DataLoader (REQUIRED for DP)
# -----------------------------

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=512,
    shuffle=False
)

# -----------------------------
# 4. Model (MLP)
# -----------------------------
class IncomeClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()

model = IncomeClassifier(X_train.shape[1])

# -----------------------------
# 5. Loss + Optimizer
# -----------------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 6. Make it PRIVATE (DP-SGD)
# -----------------------------
privacy_engine = PrivacyEngine()

model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier= 1.0,
    max_grad_norm=1.0
)

# -----------------------------
# 7. Training loop
# -----------------------------
epochs = 50

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    print(f"Epoch {epoch+1} | Loss {total_loss:.4f} | ε = {epsilon:.2f}")

# -----------------------------
# 8. Evaluation
# -----------------------------
model.eval()
preds = []
true = []

with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb)
        probs = torch.sigmoid(logits)
        preds.extend((probs >= 0.5).int().numpy())
        true.extend(yb.numpy())

accuracy = accuracy_score(true, preds)
print(f"DP Test Accuracy: {accuracy:.4f}")
