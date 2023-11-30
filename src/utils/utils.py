import torch

PIC_SIZE = 256

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
    