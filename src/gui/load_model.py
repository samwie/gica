import torch
import sys
sys.path.append("./../core")
from model_structure import UNet



def load_model(path):
    '''
    Load trained UNet model
    '''
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model = UNet()
    model.load_state_dict(
    torch.load(
        path,
        map_location=torch.device(device),
        )
    )
    return model