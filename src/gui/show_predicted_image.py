from tkinter import ttk
import sys

sys.path.append('./../utils')

from utils import predict

def show_predicted_image(model, image, root):
    '''
    Display image with color prediction
    '''
    global tk_image_pred
    
    tk_image_pred = predict(model, image)

    im_window = ttk.Frame(root, padding=10)
    im_window.place(relx=0.5, rely=0.1)
    ttk.Label(im_window, image=tk_image_pred).grid(column=0, row=0)
    