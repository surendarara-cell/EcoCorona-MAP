"""Training script for the EcoCorona-MAP sample workflow."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import EcoCoronaRegressor


def main() -> None:
    data_path = Path("data/sample_protein_np_pairs.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    feature_cols = [
        "molecular_weight_kda",
        "pi",
        "gravy",
        "net_charge_ph72",
        "hydrophobic_ratio",
        "polar_ratio",
        "surface_area",
        "zeta_potential",
    ]
    target_col = "adsorption_energy"

    x = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    x_train_tensor = torch.from_numpy(x_train)
    y_train_tensor = torch.from_numpy(y_train)
    x_test_tensor = torch.from_numpy(x_test)

    model = EcoCoronaRegressor(input_dim=x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        prediction = model(x_train_tensor)
        loss = criterion(prediction, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_prediction = model(x_test_tensor).numpy()

    rmse = mean_squared_error(y_test, test_prediction, squared=False)
    print(f"Test RMSE: {rmse:.4f}")

    Path("weights").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "weights/ecocorona_sample_model.pt")
    print("Saved model weights to weights/ecocorona_sample_model.pt")


if __name__ == "__main__":
    main()
