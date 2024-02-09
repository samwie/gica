from PIL import Image

from .select_image import select_image

from ..utils.utils import load_model, predict
from ..utils.setup_logger import logger

def return_image(path: str) -> Image.Image:
    '''
    Load a pre-trained model, select and process an image, and return the predicted color image.
    '''

    model = load_model('./../trained_model.pth')
    image = select_image(path)
    pred_image = predict(model, image)

    logger.info('The generated image was returned')

    return pred_image