import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import glob

import argparse

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from make_dataloaders import make_dataloaders
from model_structure import UNet

parser = argparse.ArgumentParser(prog="TraningLoop")
parser.add_argument(
    "-d",
    "--dir",
    dest="data_path",
    default="./../../dataset/*.jpg",
    help="Path to your dataset",
)
arguments = parser.parse_args()


def train_model(model, train_dl, epochs, learning_rate=0.001):
    criterion = nn.MSELoss()  # Funkcja straty Mean Squared Error
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, data in tqdm(enumerate(train_dl), total=len(train_dl)):
            inputs, targets = data["L"], data["ab"]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dl)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")


# set_dir = glob.glob("./../../dataset/*.jpg")
set_dir = glob.glob(arguments.data_path)

train_set = set_dir[:5703]

train_dl = make_dataloaders(paths=train_set)


model = UNet().to("cpu")
train_model(model, train_dl, epochs=100, learning_rate=0.0002)
