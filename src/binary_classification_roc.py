import torch
import torch.nn as nn
import torch.optim as optim
from data_preparation import load_california_housing
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# --- Load California housing dataset ---
X_train, X_test, y_train, y_test = load_california_housing()

# --- Function to create binary labels with a given threshold ---
def create_binary_labels(y, threshold):
    return (y > threshold).float()

# --- Set a threshold (median by default) and prepare binary labels ---
threshold = torch.median(y_train)  # You can change this value
y_train_bin = create_binary_labels(y_train, threshold)
y_test_bin = create_binary_labels(y_test, threshold)

# --- Logistic Regression model ---
class BinaryHouseClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)  # 2 features input, 1 output

    def forward(self, x):
        return self.linear(x)

model = BinaryHouseClassifier()

# --- Loss function and optimizer ---
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy with logits
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --- Training loop ---
epochs = 500
for epoch in range(epochs):
    logits = model(X_train)
    loss = criterion(logits, y_train_bin)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# --- Evaluation and ROC curve ---
with torch.no_grad():
    logits_test = model(X_test)
    probs = torch.sigmoid(logits_test)  # Convert logits to probabilities
    preds = (probs >= 0.5).float()      # Threshold at 0.5
    accuracy = (preds == y_test_bin).float().mean()
    print("Binary Accuracy:", accuracy.item())

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test_bin.numpy(), probs.numpy())
    roc_auc = auc(fpr, tpr)
    print("AUC:", roc_auc)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
