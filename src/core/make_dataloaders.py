import sys
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.append('./../utils')
from utils.constants import PIC_SIZE

class ColorizationDataset(Dataset):
    '''Dataset class designed for image colorization tasks
    '''
    def __init__(self, paths):
        self.transforms = transforms.Resize((PIC_SIZE, PIC_SIZE), Image.BICUBIC)
        self.pic_size = PIC_SIZE
        self.paths = paths

    def __getitem__(self, idx):
        '''Retrieves and processes the image
        '''
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50.0 - 1.0
        ab = img_lab[[1, 2], ...] / 110.0

        return {"L": L, "ab": ab}

    def __len__(self):
        '''Return the total number of images in the dataset
        '''
        return len(self.paths)


def make_dataloaders(batch_size=16, n_workers=1, pin_memory=True, **kwargs):
    '''Create and return dataloader
    '''
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=pin_memory
    )

    return dataloader
