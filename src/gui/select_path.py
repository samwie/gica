from tkinter import filedialog, ttk, Tk
from typing import Optional
import cv2
from PIL import Image, ImageTk
from numpy.typing import NDArray

from ..utils.setup_logger import logger

def select_path(root: Tk) -> Optional[NDArray]:
    '''
    Select the path to the image and process it
    '''
    global tk_image
    global image

    try:
        path = filedialog.askopenfilename()
        if path:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if image is not None and not image.size == 0:
                
                image = cv2.resize(image, (256, 256))
                image_pil = Image.fromarray(image)

                im_window   = ttk.Frame(root, padding=10)
                im_window.place(relx=0, rely=0.1)
                tk_image = ImageTk.PhotoImage(image_pil)
                ttk.Label(im_window, image=tk_image).grid(column=0, row=0)

                logger.info('The selected image has been successfully loaded')

                return image
            else:
                logger.warning('Unable to load the image.')
        else:
            logger.warning('Unable to load the image.')

    except FileNotFoundError:
        logger.error('File not found')
