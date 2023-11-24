from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import filedialog
import torch
import cv2
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from torchvision import transforms
import numpy as np

import sys

from ..core.model_structure import UNet

sys.path.append(
    "/home/samuel/Dokumenty/Programy/Grayscale-image-coloring-application/model_training_section"
)

root = Tk()

root.title("Image coloring application")
root.geometry("600x400")

model = UNet()
model.load_state_dict(
    torch.load(
        "/home/samuel/Dokumenty/Programy/Grayscale-image-coloring-application/model_training_section/trained_model.pth",
        map_location=torch.device("cpu"),
    )
)


global image


def select_path():
    global tk_image
    global image

    path = filedialog.askopenfilename()
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # image = cv2.resize(image, (256, 256))
    image_pil = Image.fromarray(image)

    im_window = ttk.Frame(root, padding=10)
    im_window.place(relx=0, rely=0.1)
    tk_image = ImageTk.PhotoImage(image_pil)
    ttk.Label(im_window, image=tk_image).grid(column=0, row=0)


def predict():
    global image
    global tk_image_pred

    img_normalized = image / 50.0 - 1
    img_tensor = torch.tensor(img_normalized).float().unsqueeze(0).unsqueeze(0)
    image_pred = model.predict(img_tensor)
    tk_image_pred = ImageTk.PhotoImage(image_pred)
    im_window = ttk.Frame(root, padding=10)

    im_window.place(relx=0.5, rely=0.1)
    ttk.Label(im_window, image=tk_image_pred).grid(column=0, row=0)


# buttons
btn_1 = Button(root, text="Load image", fg="black", command=select_path)
btn_1.place(relx=0.2, rely=0.01)

btn_2 = Button(root, text="Predict", fg="black", command=predict)
btn_2.place(relx=0.7, rely=0.01)


root.mainloop()

if __name__ == "__main__":
    main()