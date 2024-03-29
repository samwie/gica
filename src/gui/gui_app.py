from .select_path import select_path

from ..utils.utils import load_model

from .gui_builder import GuiBuilder

from .show_predicted_image import show_predicted_image

from ..utils.setup_logger import logger

class Gui_window(GuiBuilder):

    def __init__(self, title: str, width: int, height: int, path: str):
        super().__init__(title, width, height)
        self.image = None
        self.model = load_model(path)
        
        btn_1 = GuiBuilder.CreateButton(text="Load image", fg="black", command=self.load_image, relx=0.2, rely=0.02)
        btn_2 = GuiBuilder.CreateButton(text="Predict", fg="black", command=self.prediction, relx=0.7, rely=0.02)

    def load_image(self) -> None:
        '''Opens a file dialog to select and load an image
        '''
        self.image = select_path(self.root)

    def prediction(self) -> None:
        '''Performs colorization prediction on the loaded image using the pre-trained model
        '''
        if self.image is not None:
            show_predicted_image(self.model, self.image, self.root)



def gui_instance():
    '''
    Create and run the GUI
    '''
    path = './../trained_model.pth'
    gui = Gui_window(title=  "Image coloring application", width = 600, height = 320, path = path)
    gui.run()
