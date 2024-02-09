import logging

logger = logging.getLogger(__name__)

formatter= logging.Formatter('%(asctime)s~%(levelname)s~%(message)s~module:%(module)s')
filename='logfile.log'

logger.setLevel(logging.DEBUG)


console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(filename)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)