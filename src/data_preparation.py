# data_preparation.py
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_california_housing(n_samples=1000, features=["MedInc", "AveRooms"]):
    cal_housing = fetch_california_housing()
    X = cal_housing.data
    y = cal_housing.target.reshape(-1, 1)

    feat_idx = [cal_housing.feature_names.index(f) for f in features]
    X = X[:, feat_idx]

    # small sampel for higher speed
    X = X[:n_samples]
    y = y[:n_samples]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test

