import numpy as np
import torch
import cv2

from blazeface import BlazeFace

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def save_detections(img, detections, output_path="output.jpg"):
    # Convert tensor to numpy if needed
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    # Ensure detections are 2D
    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    # Make a copy of the image (so original isn't modified)
    img_out = img.copy()

    h, w = img.shape[:2]

    print("Found %d faces" % detections.shape[0])

    for i in range(detections.shape[0]):
        ymin = int(detections[i, 0] * h)
        xmin = int(detections[i, 1] * w)
        ymax = int(detections[i, 2] * h)
        xmax = int(detections[i, 3] * w)

        # Draw rectangle (red box)
        cv2.rectangle(
            img_out,
            (xmin, ymin),
            (xmax, ymax),
            color=(0, 0, 255),  # BGR: red
            thickness=2
        )

    # Save image
    cv2.imwrite(output_path, img_out)

gpu = "cpu"

back_net = BlazeFace(back_model=True).to(gpu)
back_net.load_weights("blazefaceback.pth")
back_net.load_anchors("anchorsback.npy")

back_net.min_score_thresh = 0.75
back_net.min_suppression_threshold = 0.3

img = cv2.imread("messi.webp")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = cv2.copyMakeBorder(
    cv2.resize(img, (int(img.shape[1] * min(256/img.shape[1], 256/img.shape[0])),
                     int(img.shape[0] * min(256/img.shape[1], 256/img.shape[0])))),
    top=(256 - int(img.shape[0] * min(256/img.shape[1], 256/img.shape[0]))) // 2,
    bottom=(256 - int(img.shape[0] * min(256/img.shape[1], 256/img.shape[0]))) - (256 - int(img.shape[0] * min(256/img.shape[1], 256/img.shape[0]))) // 2,
    left=(256 - int(img.shape[1] * min(256/img.shape[1], 256/img.shape[0]))) // 2,
    right=(256 - int(img.shape[1] * min(256/img.shape[1], 256/img.shape[0]))) - (256 - int(img.shape[1] * min(256/img.shape[1], 256/img.shape[0]))) // 2,
    borderType=cv2.BORDER_CONSTANT,
    value=[0, 0, 0]
)

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
back_detections = back_net.predict_on_image(img)
print(back_detections.shape)
print(back_detections)

save_detections(img, back_detections)