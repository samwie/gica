import sys

from src.console.console_app import console_instance
from src.gui.gui_app import gui_instance
from src.utils.setup_logger import logger

def main() -> None:

    try:

        if len(sys.argv) > 1:
            logger.info('The console app has been run')
            console_instance()

        else:
            logger.info('The GUI instance has been created')
            gui_instance()

    except Exception as e:
        logger.error(f'Error: {e}')

if __name__ == "__main__":
    main()