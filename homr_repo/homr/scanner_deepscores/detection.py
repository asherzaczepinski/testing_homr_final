from sahi.predict import get_sliced_prediction
from sahi.prediction import PredictionResult, ObjectPrediction
from sahi.postprocess.combine import GreedyNMMPostprocess
from sahi.utils.cv import visualize_object_predictions
from sahi.annotation import BoundingBox
from homr.scanner_deepscores.yolo10_sahi_detection_model import Yolov10DetectionModel
import torch
import numpy as np
import os
from functools import cmp_to_key
from typing import List, Optional

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_weights = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'weights/best.pt')

# Lazy initialization to avoid loading model at import time
_detection_model = None

def get_detection_model():
    global _detection_model
    if _detection_model is None:
        _detection_model = Yolov10DetectionModel(
            model_path=model_weights,
            confidence_threshold=0.5,
            device="cpu", # 'cpu' or 'cuda:0'
        )
    return _detection_model

def detect_everything(
    source: np.array
):
    """detect objects

    Args:
        source (np.array): image

    Returns:
        PredictionResult: image and prediction_list
    """
    detection_model = get_detection_model()
    result = get_sliced_prediction(
        source,
        detection_model,
        slice_height = 640,
        slice_width = 640,
        overlap_height_ratio = 0.2,
        overlap_width_ratio = 0.5,
        postprocess_type = "NMM", # "GREEDYNMM" "NMM" "NMS"
        postprocess_match_metric = "IOS",
        postprocess_match_threshold = 0.1,
    )

    return result

def visualize_predictions(
    image: np.array, 
    predictions_list,
    filter = None,
    text_size: float = None,
    rect_th: int = None,
    hide_labels: bool = False,
    hide_conf: bool = False
):
    """visualize image 

    Args:
        image (np.array): image with detected objects
        predictions_list (_type_): list of detected objects
        filter (_type_, optional): what classes to visualize. Defaults to None.
        text_size (float, optional): size of the category name over box. Defaults to None.
        rect_th (int, optional): rectangle thickness. Defaults to None.
        hide_labels (bool, optional): hide labels. Defaults to False.
        hide_conf (bool, optional): hide confidence. Defaults to False.

    Returns:
        np.array: image with bboxes
    """
    if filter:
        predictions_list = filter_predictions(predictions_list, filter)

    im_with_det = visualize_object_predictions(
        image=np.ascontiguousarray(image),
        object_prediction_list=predictions_list,
        rect_th=rect_th,
        text_size=text_size,
        text_th=None,
        color=None,
        hide_labels=hide_labels,
        hide_conf=hide_conf
    )
    
    return im_with_det["image"]

def filter_predictions(prediction_list, classes_to_view):
    filtered_list = []
    for prediction in prediction_list:
        if prediction.category.name in classes_to_view:
            filtered_list.append(prediction)
    return filtered_list

def compare_boxes_vertically(obj1:ObjectPrediction, obj2:ObjectPrediction):
    return obj1.bbox.to_xyxy()[1] - obj2.bbox.to_xyxy()[1]

def compare_boxes_horizontally(obj1:ObjectPrediction, obj2:ObjectPrediction):
    return obj1.bbox.to_xyxy()[0] - obj2.bbox.to_xyxy()[0]

def get_mid_lines(sorted_staffs:List[ObjectPrediction]):
    lines = []
    for i in range(len(sorted_staffs)-1):
        lines.append((sorted_staffs[i].bbox.to_xyxy()[3] + sorted_staffs[i+1].bbox.to_xyxy()[1]) / 2)
    return lines

def get_slice_bbox(lines, image_size):
    w, h = image_size
    y_min = 0
    # [x_min, y_min, x_max, y_max]
    slice_bboxes = []
    for line in lines:
        slice_bboxes.append([0, int(y_min), w, int(line)])
        y_min = int(line)
    slice_bboxes.append([0, y_min, w, h])
    return slice_bboxes

def prediction_inside_slice(prediction: ObjectPrediction, slice_bbox: List[int]) -> bool:
    """Check whether prediction coordinates lie inside slice coordinates.

    Args:
        prediction (dict): Single prediction entry in COCO format.
        slice_bbox (List[int]): Generated from `get_slice_bbox`.
            Format for each slice bbox: [x_min, y_min, x_max, y_max].

    Returns:
        (bool): True if any annotation coordinate lies inside slice.
    """
    left, top, right, bottom = prediction.bbox.to_xyxy()

    if right <= slice_bbox[0]:
        return False
    if bottom <= slice_bbox[1]:
        return False
    if left >= slice_bbox[2]:
        return False
    if top >= slice_bbox[3]:
        return False
    
    return True

def transform_prediction(prediction: ObjectPrediction, slice_bbox: List[int]):
    # prediction.bbox.miny = prediction.bbox.miny - slice_bbox[1]
    # prediction.bbox.maxy = prediction.bbox.maxy - slice_bbox[1]    
    bbox = prediction.bbox
    # construct a new bbox object
    new_bbox = BoundingBox(
        [bbox.minx,
        bbox.miny - slice_bbox[1],
        bbox.maxx,
        bbox.maxy - slice_bbox[1]]
    )
    prediction.bbox = new_bbox


def postprocess(data, match_threshold=0.1, match_metric="IOU", class_agnostic=False):
    postprocess = GreedyNMMPostprocess(
        match_threshold=match_threshold,
        match_metric=match_metric,
        class_agnostic=class_agnostic,
    )
    return postprocess(data)

def slice_image(predicted_image: PredictionResult, divider:str):
    # staffs will be sorted later
    sorted_staffs = filter_predictions(predicted_image.object_prediction_list, set([divider]))

    # merge staffs if there are intersections
    # postprocess = GreedyNMMPostprocess(
    #     match_threshold=0.1,
    #     match_metric="IOS",
    #     class_agnostic=False,
    # )

    # sort vertically
    if len(sorted_staffs) > 1:
        sorted_staffs = sorted(postprocess(sorted_staffs, match_metric="IOS"), key=cmp_to_key(compare_boxes_vertically))

    # get slice bboxes each with only one staff
    lines = get_mid_lines(sorted_staffs)
    slices = get_slice_bbox(lines, predicted_image.image.size)

    sliced_images = []
    for slice in slices:
        # extract image
        tlx = slice[0]
        tly = slice[1]
        brx = slice[2]
        bry = slice[3]
        slice_objects = []
        for object in predicted_image.object_prediction_list:
            # transform prediction coordinates if it is in the slice
            if prediction_inside_slice(object, slice):
                transform_prediction(object, slice)
                slice_objects.append(object)
        sliced_images.append({'image': np.asarray(predicted_image.image)[tly:bry, tlx:brx], 'predictions': slice_objects})
    return sliced_images