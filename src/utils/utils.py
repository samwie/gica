import torch
import sys

from ..core.model_structure import UNet

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
    try:

        device = check_cuda_availability()
        
        model = UNet()
        model.load_state_dict(
        torch.load(
            path,
            map_location=torch.device(device),
            )
        )
        return model

    except FileNotFoundError:
        print (f'Error: File not found')

    except Exception as e:
        print(f'Unexpected error: {e}')

def predict(model, image):
    '''
    Image color prediction
    '''

    img_normalized = image / 50.0 - 1
    img_tensor = torch.tensor(img_normalized).float().unsqueeze(0).unsqueeze(0)

    try:
        image_pred = model.predict(img_tensor)
        return image_pred

    except torch.TensorError as e:
        print (f'Tensor error: {e}')

    except Exception as e:
        print(f'Error: {e}')