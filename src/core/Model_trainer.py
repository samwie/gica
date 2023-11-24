import glob
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from make_dataloaders import make_dataloaders

class  Model_trainer():
    def __init__(self, model, epochs: int, set_dir: str, learning_rate:  float):
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        self.set_dir = set_dir
        self.train_set = glob.glob(set_dir)[:5703]
        self.train_dl = make_dataloaders(paths=self.train_set)

        
    def check_cuda_availability(self):
        if torch.cuda.is_available():
            print ('CUDA is available and set as device')
            device = 'cuda'
        else:
            print('CUDA is not available. CPU is set as device')
            device = 'cpu'
        return device
    
    def train_model(self):
        criterion = nn.MSELoss()  # Loss function Mean Squared Error*.jpg
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        device = self.check_cuda_availability()
        self.model.to(device)

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            for batch_idx, data in tqdm(enumerate(self.train_dl), total=len(self.train_dl)):
                inputs, targets = data["L"], data["ab"]

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(self.train_dl)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}")