from PIL import Image
import numpy as np
import os
from pdf2image import convert_from_path, convert_from_bytes


def load_test_image():
    file_name = 'Poker Face Fl 2.pdf'
    # file_name = 'lg-2267728-aug-beethoven--page-2.png' 
    test_im_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'test_data', file_name)
    img_file = convert_from_path(test_im_path)[0]
    test_image = np.array(img_file)
    return test_image

def get_photo(file, key):
    bytes_data = file.getvalue()
    file_type = file.type.split('/')[-1]
    if file_type=='pdf':
        #TODO add multiple pages    
        img_file = convert_from_bytes(bytes_data)[0]
    else:
         img_file = Image.open(file)
    # convert PIL Image to numpy array:
    notes_image = np.array(img_file)

    return notes_image
