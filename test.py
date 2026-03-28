import numpy as np
import torch
import cv2

from blazeface import BlazeFace

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def save_detections_on_original(
    original_img,
    detections,
    scale,
    pad_top,
    pad_left,
    resized_shape,
    output_path="output.jpg"
):
    if detections.ndim == 1: detections = np.expand_dims(detections, axis=0)

    img_out = original_img.copy()
    orig_h, orig_w = original_img.shape[:2]
    resized_h, resized_w = resized_shape

    print("Found %d faces" % detections.shape[0])

    for i in range(detections.shape[0]):
        ymin = detections[i, 0] * resized_h
        xmin = detections[i, 1] * resized_w
        ymax = detections[i, 2] * resized_h
        xmax = detections[i, 3] * resized_w

        ymin -= pad_top
        ymax -= pad_top
        xmin -= pad_left
        xmax -= pad_left

        ymin /= scale
        ymax /= scale
        xmin /= scale
        xmax /= scale

        x1, y1, x2, y2 = map(int, [xmin, ymin, xmax, ymax])

        x1 = max(0, min(orig_w, x1))
        x2 = max(0, min(orig_w, x2))
        y1 = max(0, min(orig_h, y1))
        y2 = max(0, min(orig_h, y2))

        cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(output_path, img_out)

gpu = "cpu"

back_net = BlazeFace(back_model=True).to(gpu)
back_net.load_weights("blazefaceback.pth")
back_net.load_anchors("anchorsback.npy")

back_net.min_score_thresh = 0.75
back_net.min_suppression_threshold = 0.3

orig = cv2.imread("messi.webp")
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

h0, w0 = orig.shape[:2]

scale = min(256 / w0, 256 / h0)
new_w, new_h = int(w0 * scale), int(h0 * scale)

resized = cv2.resize(orig, (new_w, new_h))

pad_top = (256 - new_h) // 2
pad_bottom = (256 - new_h) - pad_top
pad_left = (256 - new_w) // 2
pad_right = (256 - new_w) - pad_left

img = cv2.copyMakeBorder(
    resized,
    pad_top, pad_bottom, pad_left, pad_right,
    borderType=cv2.BORDER_CONSTANT,
    value=[0, 0, 0]
)

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

detections = back_net.predict_on_image(img).numpy()
print(detections)

s = "["
for y in detections:
    s += "["
    for x in y: s += str(x) + ","
    s += "],\n"
print(s,"]")

expected = [[0.22293027,0.3687327,0.35492355,0.500726,0.4048541,0.253551,0.45936358,0.25396332,0.42835188,0.2809909,0.42859644,0.31245646,0.37655264,0.27385083,0.49636966,0.27672035,0.83855903,],
[0.30805102,0.68929595,0.42866126,0.8099063,0.71050656,0.34094658,0.75901216,0.34136337,0.7211923,0.3699867,0.7258061,0.3949228,0.703986,0.3506133,0.8086657,0.3542543,0.7997207,],]

np.testing.assert_allclose(detections, expected, rtol=1e-6, atol=1e-6)

save_detections_on_original(
    original_img=cv2.cvtColor(orig, cv2.COLOR_RGB2BGR),
    detections=detections,
    scale=scale,
    pad_top=pad_top,
    pad_left=pad_left,
    resized_shape=(256, 256),
    output_path="result.jpg"
)