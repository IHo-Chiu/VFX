import cv2
import numpy as np

def nonmaximum_suppression(image, thres, window_size = 5):
    keypoints = []
    extreme = True
    for x in range(window_size // 2 + 1, image.shape[0] - window_size // 2 - 2):
        for y in range(window_size // 2 + 1, image.shape[1] - window_size // 2 - 2):
            if thres[x][y] == 255:
                extreme = True
            for i in range(-window_size // 2, window_size // 2):
                for j in range(-window_size // 2, window_size // 2):
                    if image[x+i][y+j] > image[x][y]:
                        extreme = False
                    if extreme == False:
                        break
                if extreme == False:
                    break
            if extreme == True:
                keypoints.append([y, x])

    return keypoints

def Harris_detector(image, k = 0.06):

    # Compute x and y derivatives of image
    iy, ix = np.gradient(image)

    # Compute products of derivatives at every pixe
    ix2 = ix * ix
    iy2 = iy * iy
    ixy = ix * iy

    # Compute the sums of the products of derivatives at each pixel
    sx2 = cv2.GaussianBlur(ix2 , (0 , 0) , sigmaX = 1.6)
    sy2 = cv2.GaussianBlur(iy2 , (0 , 0) , sigmaX = 1.6)
    sxy = cv2.GaussianBlur(ixy , (0 , 0) , sigmaX = 1.6)

    # Define the matrix at each pixel
    detM = (sx2 * sy2) - (sxy ** 2)
    traceM = sx2 + sy2
    R = detM - k * (traceM ** 2)

    # Threshold on value of R
    _, binary_r = cv2.threshold(R.astype("uint8"), 0, 255, cv2.THRESH_OTSU)

    keypoints = nonmaximum_suppression(R, binary_r)
    return keypoints, ix, iy, ix2, iy2
