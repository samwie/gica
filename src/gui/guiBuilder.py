import tkinter as tk

class GuiBuilder:
    '''
    GUI builder class using tkinter
    '''
    def __init__ (self, title: str, width: int, height: int):
        """
        Initializes the GuiBuilder object.

        Parameters:
            title (str): The title of the GUI window.
            width (int): The width of the GUI window.
            height (int): The height of the GUI window.
        """
        self.root  = tk.Tk()
        self.root.geometry(f'{width}x{height}')
        self.root.title(title)

    @property
    def run(self):
        """
        Runs the tkinter main loop to display the GUI.
        """
        self.root.mainloop()

    def CreateButton(text: str, fg: str, command: callable, relx: float, rely: float):
        """
        Creates and places a button in the GUI window
        Parameters:
            text (str): The text displayed on the button.
            fg (str): The foreground color of the button text.
            command (callable): The function to be called when the button is clicked.
            relx (float): The relative x-coordinate for placing the button.
            rely (float): The relative y-coordinate for placing the button.
        """
        btn = tk.Button(text = text, fg=fg, command=command)
        btn.place(relx=relx, rely=rely)

        return btn
    