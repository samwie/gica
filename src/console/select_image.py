import cv2
from typing import Optional
from numpy.typing import NDArray

from ..utils.setup_logger import logger

def select_image(path: str) -> Optional[NDArray]:
    '''
    Load and preprocess a grayscale image.
    '''
    try:

        if path:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if image is not None and not image.size == 0:

                image = cv2.resize(image, (256, 256))

                logger.info('Image was selected')

                return image

            else:
                logger.warning('Unable to load the image.')

            logger.warning('No file selected')

    except FileNotFoundError:
        logger.warning('File not found')

    except cv2.error as e:
        logger.error(f'OpenCV error: {e}')
