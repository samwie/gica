from tkinter import ttk
import torch
from PIL import ImageTk


def predict(model, image, root):
    '''
    Image color prediction
    '''
    global tk_image_pred

    img_normalized = image / 50.0 - 1
    img_tensor = torch.tensor(img_normalized).float().unsqueeze(0).unsqueeze(0)
    image_pred = model.predict(img_tensor)
    tk_image_pred = ImageTk.PhotoImage(image_pred)
    im_window = ttk.Frame(root, padding=10)

    im_window.place(relx=0.5, rely=0.1)
    ttk.Label(im_window, image=tk_image_pred).grid(column=0, row=0)
