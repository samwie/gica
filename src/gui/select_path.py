from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk


def select_path(root):
    '''
    Select the path to the image and process it
    '''
    global tk_image
    global image

    try:
        path = filedialog.askopenfilename()
        if path:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if image is not None and not image.size == 0:
                
                image = cv2.resize(image, (256, 256))
                image_pil = Image.fromarray(image)

                im_window   = ttk.Frame(root, padding=10)
                im_window.place(relx=0, rely=0.1)
                tk_image = ImageTk.PhotoImage(image_pil)
                ttk.Label(im_window, image=tk_image).grid(column=0, row=0)

                return image
            else:
                print("Error: Unable to load the image.")
        else:
            print("Error: Unable to load the image.")

    except FileNotFoundError:
        print('Error: File not found')

    except Exception as e:
        print(f'Unexpected error: {e}')
        
