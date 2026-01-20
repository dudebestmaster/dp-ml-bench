from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# ============================
# 1. Load dataset
# ============================
adult = fetch_ucirepo(id=2)

X = adult.data.features.copy()
y = adult.data.targets.copy()


# ============================
# 2. Clean labels
# ============================
y = y.iloc[:, 0]
y = y.str.strip()
y = y.str.replace('.', '', regex=False)
y = (y == ">50K").astype(int).values


# ============================
# 3. Remove missing values
# ============================
X = X.replace('?', np.nan)
mask = X.notna().all(axis=1)

X = X.loc[mask]
y = y[mask]


# ============================
# 4. One-hot encode (DENSE)
# ============================
X = pd.get_dummies(X, drop_first=True)


# ============================
# 5. Train / test split (STRATIFIED)
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================
# 6. Scale features (DENSE)
# ============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================
# 7. Convert to PyTorch tensors
# ============================
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# ============================
# 8. MODEL: Small MLP
# ============================
class IncomeMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()


model = IncomeMLP(X_train.shape[1])


# ============================
# 9. Loss + optimizer
# ============================
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)


# ============================
# 10. Training loop
# ============================
epochs = 50

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    logits = model(X_train)
    loss = criterion(logits, y_train)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:02d} | Loss {loss.item():.4f}")


# ============================
# 11. Evaluation
# ============================
model.eval()
with torch.no_grad():
    logits = model(X_test)
    preds = (torch.sigmoid(logits) >= 0.5).int()
    acc = accuracy_score(y_test.numpy(), preds.numpy())

print(f"\nBaseline Test Accuracy: {acc:.4f}")
