import numpy as np
import torch
import cv2

from blazeface import BlazeFace

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_detections(img, detections, with_keypoints=True):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.grid(False)
    ax.imshow(img)
    
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()

    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    print("Found %d faces" % detections.shape[0])
        
    for i in range(detections.shape[0]):
        ymin = detections[i, 0] * img.shape[0]
        xmin = detections[i, 1] * img.shape[1]
        ymax = detections[i, 2] * img.shape[0]
        xmax = detections[i, 3] * img.shape[1]

        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 linewidth=1, edgecolor="r", facecolor="none", 
                                 alpha=detections[i, 16])
        ax.add_patch(rect)

        if with_keypoints:
            for k in range(6):
                kp_x = detections[i, 4 + k*2    ] * img.shape[1]
                kp_y = detections[i, 4 + k*2 + 1] * img.shape[0]
                circle = patches.Circle((kp_x, kp_y), radius=0.5, linewidth=1, 
                                        edgecolor="lightskyblue", facecolor="none", 
                                        alpha=detections[i, 16])
                ax.add_patch(circle)
        
    plt.show()

gpu = "cpu"

back_net = BlazeFace(back_model=True).to(gpu)
back_net.load_weights("blazefaceback.pth")
back_net.load_anchors("anchorsback.npy")

# Optionally change the thresholds:
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


back_detections = back_net.predict_on_image(img)
print(back_detections.shape)
print(back_detections)

plot_detections(img, back_detections)