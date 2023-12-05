import sys

from select_image import select_image

sys.path.append('./../utils')
from utils import load_model, predict

def return_image(path):
    '''
    Load a pre-trained model, select and process an image, and return the predicted color image.
    '''
    model = load_model('./../../../trained_model.pth')
    image = select_image(path)
    pred_image = predict(model, image)

    return pred_image