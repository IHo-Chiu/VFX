
import csv
import numpy as np
from PIL import Image
import cv2

def load_images(csv_path, return_images_name=False, shifted=False):
    image_paths = []
    exposures = []
    with open(csv_path, newline='') as csvfile:
        rows = csv.reader(csvfile)
        for i, row in enumerate(rows):
            if i == 0: continue
            image_paths.append(row[0])
            exposures.append(float(row[1]))

    images = []
    for image_path in image_paths:
        if shifted:
            image_path = 'shifted_'+image_path
        img = np.asarray(Image.open('data/'+image_path))
        images.append(img)

    if return_images_name:
        return images, exposures, image_paths
    else:
        return images, exposures

def down_sample(images, scale=32):
    new_images = []
    for image in images:
        h, w = image.shape
        res = cv2.resize(image, dsize=(w//scale, h//scale), interpolation=cv2.INTER_CUBIC)
        new_images.append(res)

    return new_images
