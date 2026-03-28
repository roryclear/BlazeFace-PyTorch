import numpy as np
import torch
import cv2

from blazeface import BlazeFace

gpu = "cpu"

front_net = BlazeFace().to(gpu)
front_net.load_weights("blazeface.pth")
front_net.load_anchors("anchors.npy")
back_net = BlazeFace(back_model=True).to(gpu)
back_net.load_weights("blazefaceback.pth")
back_net.load_anchors("anchorsback.npy")

# Optionally change the thresholds:
front_net.min_score_thresh = 0.75
front_net.min_suppression_threshold = 0.3

img = cv2.imread("1face.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

front_detections = front_net.predict_on_image(img)
print(front_detections.shape)
print(front_detections)
