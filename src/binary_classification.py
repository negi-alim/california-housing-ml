import torch
import torch.nn as nn
import torch.optim as optim

# Import تابع آماده‌سازی دیتا
from data_preparation import load_california_housing

# --- آماده‌سازی داده‌ها ---
X_train, X_test, y_train, y_test = load_california_housing()

# --- تعریف کلاس‌ها ---
# Binary labels: 1 = Luxury, 0 = Non-Luxury
threshold = torch.median(y_train)
y_train_bin = (y_train > threshold).float()
y_test_bin = (y_test > threshold).float()

# --- مدل Logistic Regression (Binary) ---
class BinaryHouseClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

model = BinaryHouseClassifier()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --- آموزش مدل ---
epochs = 500
for epoch in range(epochs):
    logits = model(X_train)
    loss = criterion(logits, y_train_bin)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# --- ارزیابی ---
with torch.no_grad():
    logits_test = model(X_test)
    probs = torch.sigmoid(logits_test)
    preds = (probs >= 0.5).float()
    accuracy = (preds == y_test_bin).float().mean()

print(f"Binary classification accuracy: {accuracy.item():.4f}")
