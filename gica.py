import sys

from src.console.console_app import console_instance
from src.gui.gui_app import gui_instance

def main():

    try:
        if len(sys.argv) > 1:
            console_instance()
        else:
            gui_instance()

    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    main()