import cv2
import numpy as np

def local_image_descriptor(image, keypoints, ix, iy, ix2, iy2):
    # Compute x and y derivatives of image
    m = (ix2 + iy2) **(1/2)
    theta = (np.arctan2(iy, ix) * 180 / np.pi + 180) % 360

    # calculate bin map
    theta_bins = theta // 45 % 8
    bin_map = np.zeros((8,) + ix.shape)
    for b in range(8):
        bin_map[b][theta_bins == b] = 1
        bin_map[b] *= m

    descriptors = []
    for (x, y) in keypoints:
        # boundary handling
        x = int(max(8, min(x, image.shape[0] - 9)))
        y = int(max(8, min(y, image.shape[1] - 9)))
        # rotate bin map
        M = cv2.getRotationMatrix2D((8, 8), theta[y, x], 1)
        ori_rotated = [cv2.warpAffine(bm[y-8:y+8, x-8:x+8], M, (16, 16)) for bm in bin_map]
        # make bin histogram and flatten
        descriptors.append(np.array(ori_rotated).reshape(8, 4, 4, 4, 4).sum(axis=(2, 4)).flatten())
    
    return np.array(descriptors)


