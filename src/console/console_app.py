import os
import sys

sys.path.append("./../utils")
import argparse

from .return_image import return_image
from ..utils.utils import load_model, predict
from ..utils.constants import DESCRIPTION, EPILOG, PATH_HELP, DESTINATION_HELP
from ..utils.setup_logger import logger

def console_instance():
    '''
    Predict color for a grayscale image using a pre-trained model.
    Command-line arguments:
    -p or --path - path to the grayscale image REQUIRED
    -d or --destination - destination path for saving the predicted color imagE OPTIONAL
    '''
    parser = argparse.ArgumentParser(description=f'{DESCRIPTION}', epilog=f'{EPILOG}')
    parser.add_argument('-p', '--path', type=str, help=f'{PATH_HELP}')
    parser.add_argument('-d', '--destination', type=str, default=None , help=f'{DESTINATION_HELP}')

    args = parser.parse_args()

    if args.path:
        path = args.path

        if args.destination is None:
            directory, filename = os.path.split(args.path)
            filename_base, filename_extension = os.path.splitext(filename)
            predicted_filename = f"{filename_base}_predicted{filename_extension}"
            args.destination = os.path.join(directory, predicted_filename)
            destination = args.destination

        else:
            directory, filename = os.path.split(args.path)
            filename_base, filename_extension = os.path.splitext(filename)
            predicted_filename = f"{filename_base}_predicted{filename_extension}"
            destination = os.path.join(args.destination, predicted_filename)

        pred_image = return_image(path)
        pred_image.save(destination)
        
        print(f'The predicted image was saved in the location: {destination}')
        logger.info('The generated image was saved')

    else:
        parser.print_help()