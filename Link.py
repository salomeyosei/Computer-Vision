
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import random
from google.colab.patches import cv2_imshow

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from os import listdir, makedirs, getcwd, remove


import torch, torchvision
import os
from detectron2.structures import pairwise_iou
from os import listdir, makedirs, getcwd, remove
import matplotlib.cm as cm
from detectron2.utils.visualizer import GenericMask, _create_text_labels, ColorMode

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml")

def get_predictions(model = cfg, foldername= 'clip'):
    
    predictor = DefaultPredictor(cfg)

    clip_40_predicted = []

    for clip in sorted(listdir(f"./{foldername}/")):
        image = cv2.imread(f"./{foldername}/"+clip)
        # print(f'Predicting Image {clip[:2]} ....\n ')    
        clip_40_predicted.append(predictor(image))
    
    print('Done.') 

    return clip_40_predicted
    

class TrackerVisualizer(Visualizer):
    def draw_instance_predictions(self, predictions, track_ids):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None
                
        # set the color according to the track ids 
        colors = [cm.tab20(id_) for id_ in track_ids]
        alpha = 0.6

        labels = [f'Track {id_} {label}' for label, id_ in zip(labels,track_ids)]
        
        # increase font size
        if self._default_font_size < 20: self._default_font_size *= 1.3
        

        if self._instance_mode == ColorMode.IMAGE_BW:
            assert predictions.has("pred_masks"), "ColorMode.IMAGE_BW requires segmentations"
            self.output.img = self._create_grayscale_image(
                (predictions.pred_masks.any(dim=0) > 0).numpy()
            )

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output

def frame_tracking(predictions, number_frame = 2):

    # counter for detected object
    number_object_in_first_image = predictions[0]["instances"].get('pred_classes').size(0)

    # inititalised with the number of detected objects in the first image
    track_counter = number_object_in_first_image

    # initial the list of track_ids with the objects in the first image
    track_ids = np.arange(0, number_object_in_first_image)


    for i in range(-1,number_frame-1):
        pred1 = predictions[i]["instances"].to("cpu")
        pred2 = predictions[i+1]["instances"].to("cpu")

        objects_overlaps = pairwise_tracker(pred1, pred2)

        max_overlaps, indices = objects_overlaps.max(dim=0)

        new_track_ids = []
        for max_overlap, index in zip(max_overlaps, indices):
            if max_overlap == 0: # no track object assign new identifier
                new_track_ids.append(track_counter)
                track_counter += 1
            else:
                new_track_ids.append(track_ids[index])

        track_ids = new_track_ids
        
        # We can use `Visualizer` to draw the predictions on the image.
        filename = f'{i+1:02d}.jpg' 
        print(f"\nImage {filename[:-4]}\n")
        im = cv2.imread('./clip/'+filename)
        v = TrackerVisualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(pred2, track_ids)
        cv2_imshow(v.get_image()[:, :, ::-1])

def pairwise_tracker(pred1, pred2):
    boxes1 = pred1.get('pred_boxes')
    boxes2 = pred2.get('pred_boxes')
    
    categories1 = pred1.get('pred_classes')
    categories2 = pred2.get('pred_classes')

    boxes_overlaps = pairwise_iou(boxes1, boxes2)

    objects_overlaps = (categories1[:,None] == categories2[None,:]) * boxes_overlaps 

    return objects_overlaps
