import warnings
warnings.filterwarnings('ignore')
import sys, os

import glob
import partitura as pt
from partitura.score import Score, Part, Measure
import numpy as np
import matplotlib.pyplot as plt

import utils as utl
import detection as detector
import json
import yaml
import pickle
import cv2

from functools import cmp_to_key
from typing import List


from sahi.prediction import ObjectPrediction
from sahi.annotation import Category

clefRange = (5,11)
keyRange = (67,69)
timeRange = (12,23)
noteRange = (24,39)

def def_clef(object_cat:Category):
    if object_cat.id == 5:
        return '*clefG2'
    else:
        print('no clef')
        return ''
    
def def_key(object_cat:Category):
    # TODO add more than one key
    key = '*k['
    if object_cat.id == 67:
        key+='b-'
    elif object_cat.id == 69:
        key+='f#'
    return key+']'

def def_time(object_cat:Category):
    if object_cat.id == 16:
        return '*M4/4'
    else:
        print('add other metrics')
        return '*met(c)'

def def_note(object_cat:Category):
    # TODO define what note is it
    # TODO remove flat sign
    if object_cat.id == 24:
        return '4g'
    elif object_cat.id == 26:
        return '4a'
    if object_cat.id == 28:
        return '2g'
    elif object_cat.id == 30:
        return '2a'
    if object_cat.id == 32:
        return '1g'
    elif object_cat.id == 34:
        return '1a'

def comrehend(predictions:List[ObjectPrediction]):
    kern_data = ["**kern"]
    for prediction in predictions:
        if clefRange[0] <= prediction.category.id <= clefRange[1]:
            kern_data.append(def_clef(prediction.category))
        elif keyRange[0] <= prediction.category.id <= keyRange[1]:
            kern_data.append(def_key(prediction.category))
        elif timeRange[0] <= prediction.category.id <= timeRange[1]:
            kern_data.append(def_time(prediction.category))
            kern_data.append('=1-')
        elif noteRange[0] <= prediction.category.id <= noteRange[1]:
            kern_data.append(def_note(prediction.category))
            

    kern_data.append('*-')
    return kern_data

def save_kern(kern_data, filename):
    with open(filename, "w") as kern_file:
        kern_file.writelines('\n'.join(kern_data))

if __name__=="__main__":

    # decoded_image = utl.load_test_image()

    # detections = detector.detect_everything(decoded_image)
    # classes_to_view = set(['staff'])

    # sliced_staffs = detector.slice_image(detections, divider='staff')
    # # for staff in sliced_staffs:
    # staff = sliced_staffs[0]
    # cv2.imwrite('test_img.png', cv2.cvtColor(staff['image'], cv2.COLOR_RGB2BGR))
    # print(staff['predictions'])
    # print("--------------------------------------------")
    # with open('myfile.json', 'w', encoding ='utf8') as json_file:
    #     json.dump(staff['predictions'], json_file, ensure_ascii = False)
    # with open(r"someobject.pickle", "wb") as file:
    #     pickle.dump(staff['predictions'], file)


    # Load test image and predictions
    pickle_test_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'test_data', "someobject.pickle")
    with open(pickle_test_file, "rb") as input_file:
        predictions = pickle.load(input_file)
    image_test_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'test_data', "test_img.png")
    image = cv2.imread(image_test_file)
    sorted_predictions = sorted(predictions, key=cmp_to_key(detector.compare_boxes_horizontally))
    processed = detector.postprocess(sorted_predictions, match_metric="IOS", match_threshold=0.004)
    sorted_processed = sorted(processed, key=cmp_to_key(detector.compare_boxes_horizontally))

    print(sorted_processed)
    # visual_staff = detector.visualize_predictions(image, sorted_processed, hide_labels=True, rect_th=2, text_size=1) # , filter=classes_to_view
    # plt.imshow(visual_staff)
    # plt.show()

    # Transform detections to kern syntax
    generated_kern = comrehend(sorted_processed)

    filename = "kern_code.krn"
    filename = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'test_data', filename)
    save_kern(generated_kern, filename)

    # view generated kern_file
    new_part = pt.load_kern(filename)
    pt.render(new_part)




# **kern
# *clefG2
# *k[b-]
# *d:
# *M2/2
# *met(c)
# =1-
# 2d
# 2a
# =2
# 2f
# 2d
# =3
# 2c#
# 4d
# 4e
# =4
# [2f
# 8f]L
# 8g
# 8f
# 8eJ
# *-