import cv2

def select_image(path):
    '''
    Load and preprocess a grayscale image.
    '''
    if path:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if image is not None and not image.size == 0:

            image = cv2.resize(image, (256, 256))
            
            return image

        else:
           print("Error: Unable to load the image.")
    print("Error: No file selected.")