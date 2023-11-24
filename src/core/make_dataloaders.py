import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


PIC_SIZE = 256


class ColorizationDataset(Dataset):
    def __init__(self, paths):
        self.transforms = transforms.Resize((PIC_SIZE, PIC_SIZE), Image.BICUBIC)

        self.pic_size = PIC_SIZE
        self.paths = paths

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50.0 - 1.0
        ab = img_lab[[1, 2], ...] / 110.0

        return {"L": L, "ab": ab}

    def __len__(self):
        return len(self.paths)


def make_dataloaders(batch_size=16, n_workers=1, pin_memory=True, **kwargs):
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=pin_memory
    )

    return dataloader
