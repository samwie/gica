import sys
from tkinter import Tk

from select_path import select_path
from predict import predict
from load_model import load_model

sys.path.append("./../core")
from model_structure import UNet
from Create_Button import CreateButton

class GUI_Window:
    '''GUI window for an image coloring application
    '''
    def __init__(self, root, path):
        self.root = root
        self.root.title("Image coloring application")
        self.root.geometry("600x320")

        self.image = None
        self.model = load_model(path)

        btn_1 = CreateButton(text="Load image", fg="black", command=self.load_image, relx=0.2, rely=0.02)
        btn_2 = CreateButton(text="Predict", fg="black", command=self.prediction, relx=0.7, rely=0.02)

    def load_image(self):
        '''Opens a file dialog to select and load an image
        '''
        self.image = select_path(self.root)

    def prediction(self):
        '''Performs colorization prediction on the loaded image using the pre-trained model
        '''
        if self.image is not None:
            predict(self.model, self.image, self.root)

def gui_isinstance():
    root = Tk()
    path = "./../../../trained_model.pth"
    app = GUI_Window(root, path)
    root.mainloop()

if __name__ == "__main__":
    gui_isinstance()