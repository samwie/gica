import sys
from tkinter import Tk, Button

from select_path import select_path
from predict import predict
from load_model import load_model

sys.path.append("./../core")
from model_structure import UNet


class GUI_Window:
    def __init__(self, root, path):
        self.root = root
        self.root.title("Image coloring application")
        self.root.geometry("600x320")

        self.image = None
        self.model = load_model(path)

        btn_1 = Button(self.root, text="Load image", fg="black", command=self.load_image)
        btn_1.place(relx=0.2, rely=0.02)
        btn_2 = Button(self.root, text="Predict", fg="black", command=self.prediction)
        btn_2.place(relx=0.7, rely=0.02)
        

    def load_image(self):
        self.image = select_path(self.root)

    def prediction(self):
        if self.image is not None:
            predict(self.model, self.image, self.root)


if __name__ == "__main__":
    root = Tk()
    path = "./../../../trained_model.pth"
    app = GUI_Window(root, path)
    root.mainloop()