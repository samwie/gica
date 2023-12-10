import torch
import sys

sys.path.append("./../core")
from model_structure import UNet


def check_cuda_availability():
    '''
    Check if CUDA (GPU) available and set device respectively
    '''
    if torch.cuda.is_available():
        print ('CUDA is available and set as device')
        device = 'cuda'
    else:
        print('CUDA is not available. CPU is set as device')
        device = 'cpu'
        
    return device

def load_model(path):
    '''
    Load trained UNet model
    '''
    device = check_cuda_availability()
    
    model = UNet()
    model.load_state_dict(
    torch.load(
        path,
        map_location=torch.device(device),
        )
    )
    return model

def predict(model, image):
    '''
    Image color prediction
    '''

    img_normalized = image / 50.0 - 1
    img_tensor = torch.tensor(img_normalized).float().unsqueeze(0).unsqueeze(0)
    image_pred = model.predict(img_tensor)

    return image_pred