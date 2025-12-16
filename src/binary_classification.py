import torch
import torch.nn as nn
import torch.optim as optim

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
    loss = criterion(logits, y_train_bin)  # Compute the loss

    optimizer.zero_grad()  # Clear gradients
    loss.backward()        # Backpropagation
    optimizer.step()       # Update weights

# --- Evaluation ---
with torch.no_grad():
    logits_test = model(X_test)                 # Get logits on test set
    probs = torch.sigmoid(logits_test)         # Convert logits to probabilities
    preds = (probs >= 0.5).float()             # Threshold probabilities to get binary predictions
    accuracy = (preds == y_test_bin).float().mean()  # Compute accuracy

print(f"Binary classification accuracy: {accuracy.item():.4f}")

