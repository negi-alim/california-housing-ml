import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score, f1_score

# Import the data preparation function
from data_preparation import load_california_housing

# --- Prepare the data ---
X_train, X_test, y_train, y_test = load_california_housing()

# --- Define labels ---
# Binary labels: 1 = Luxury, 0 = Non-Luxury
threshold = torch.median(y_train)
y_train_bin = (y_train > threshold).float()
y_test_bin = (y_test > threshold).float()

# --- Logistic Regression Model (Binary) ---
class BinaryHouseClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

model = BinaryHouseClassifier()
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --- Train the model ---
epochs = 500
for epoch in range(epochs):
    logits = model(X_train)
    loss = criterion(logits, y_train_bin)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# --- Evaluation ---
with torch.no_grad():
    logits_test = model(X_test)
    probs = torch.sigmoid(logits_test)
    preds = (probs >= 0.5).float()

# Convert tensors to numpy arrays for sklearn metrics
y_true = y_test_bin.numpy()
y_pred = preds.numpy()

# Accuracy
acc = accuracy_score(y_true, y_pred)
# Recall (sensitivity for the positive class)
recall = recall_score(y_true, y_pred)
# F1-Score
f1 = f1_score(y_true, y_pred)

print(f"Binary classification metrics:")
print(f"Accuracy: {acc:.4f}")
print(f"Recall:   {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
