from tkinter import Button

def CreateButton(text, fg, command, relx, rely):
    '''
    Create and place a button in the GUI window
    '''
    btn = Button(text = text, fg=fg, command=command)
    btn.place(relx=relx, rely=rely)