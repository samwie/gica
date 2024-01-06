from tkinter import ttk, Tk
from numpy.typing import NDArray
from PIL import ImageTk

from ..utils.utils import predict

from ..utils.setup_logger import logger

def show_predicted_image(model:any, image:NDArray, root: Tk) -> None:

    '''
    Display image with color prediction
    '''

    global tk_image_pred

    tk_image_pred = ImageTk.PhotoImage(predict(model, image))

    im_window = ttk.Frame(root, padding=10)
    im_window.place(relx=0.5, rely=0.1)
    ttk.Label(im_window, image=tk_image_pred).grid(column=0, row=0)

    logger.info('The color image was displayed')
