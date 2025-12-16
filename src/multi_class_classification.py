# Thresholds for 3-class classification
low_th = torch.quantile(y_train, 0.33)
high_th = torch.quantile(y_train, 0.66)

def three_class_labels(y):
    labels = torch.zeros_like(y, dtype=torch.long)
    labels[y > low_th] = 1
    labels[y > high_th] = 2
    return labels

y_train_3 = three_class_labels(y_train).squeeze()
y_test_3 = three_class_labels(y_test).squeeze()

#Softmax mode;
class MultiClassHouseClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 3)

    def forward(self, x):
        return self.linear(x)

model_mc = MultiClassHouseClassifier()
criterion_mc = nn.CrossEntropyLoss()
optimizer_mc = optim.Adam(model_mc.parameters(), lr=0.01)

#Model training
epochs = 500
for epoch in range(epochs):
    logits = model_mc(X_train)
    loss = criterion_mc(logits, y_train_3)

    optimizer_mc.zero_grad()
    loss.backward()
    optimizer_mc.step()

#Evaluation
with torch.no_grad():
    logits_test = model_mc(X_test)
    preds = torch.argmax(logits_test, dim=1)
    accuracy = (preds == y_test_3).float().mean()
    print("3-Class Accuracy:", accuracy.item())