
import cv2
import os
import numpy as np
from PIL import Image

def down_sample(images, scale=32):
    new_images = []
    for image in images:
        h, w, c = image.shape
        res = cv2.resize(image, dsize=(w//scale, h//scale), interpolation=cv2.INTER_CUBIC)
        new_images.append(res)

    return new_images

def load_images(root):
    images = []
    img_paths = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name[-3:] not in ["JPG", "jpg", "png", "PNG"] :	
                continue
            img_path = os.path.join(path, name)
            img_paths.append(img_path)

    img_paths.sort()
    for img_path in img_paths:
        img = np.asarray(Image.open(img_path))
        images.append(img)

    return images