
import cv2
import numpy as np
from PIL import Image
from utils import down_sample, load_images
from warp import cylindricalProjection
from imageMatching import RANSAC
from imageStiching import imageStiching
from featureMatching import keypoint_matching
from featureDetecting import Harris_detector
from featureDescription import local_image_descriptor

if __name__ == "__main__":
    scale = 3
    focal_length = 3648

    # load images
    root = "../data/preprocessed"
    print(f"image_loading...", end='\r')
    images = load_images(root)
    images = down_sample(images, scale=scale)
    print(f"image_loading...done", end='\n')

    # Warp to cylindrical coordinate
    print(f"Warp to cylindrical coordinate...", end='\r')
    new_images = []
    for i in range(len(images)):
    # for i in range(2):
        print(f"Warp to cylindrical coordinate...{i}", end='\r')
        new_image = cylindricalProjection(images[i] , int(focal_length/scale))
        new_images.append(new_image)
    images = new_images
    images.append(images[0]) # end-to-end alignment
    print(f"Warp to cylindrical coordinate...done", end='\n')

    # feature decetction
    print(f"feature decetction (Harris) & description (SIFT)...", end='\r')
    kp_list = []
    des_list = []
    for i, img in enumerate(images):
        print(f"feature decetction (Harris) & description (SIFT)...{i}", end='\r')
        gray = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY)
        kp, ix, iy, ix2, iy2 = Harris_detector(gray)
        des = local_image_descriptor(gray, kp, ix, iy, ix2, iy2)
        kp_list.append(kp)
        des_list.append(des)
    print(f"feature decetction (Harris) & description (SIFT)...done", end='\n')

    # feature matching
    print(f"feature matching (exhaustive search) & (cosine similarity)...", end='\r')
    matched_kp_list = []
    for i in range(len(images)-1):
        print(f"feature matching (exhaustive search) & (cosine similarity)...{i}", end='\r')
        matched_kp = keypoint_matching(des_list[i], kp_list[i], des_list[i+1], kp_list[i+1])
        matched_kp_list.append(matched_kp)
    print(f"feature matching (exhaustive search) & (cosine similarity)...done", end='\n')

    # image matching
    print(f"image matching (RANSAC)...", end='\r')
    theta_list = []
    for i, matched_kp in enumerate(matched_kp_list):
        print(f"image matching (RANSAC)...{i}", end='\r')
        theta = RANSAC(matched_kp)
        theta_list.append(theta)
    print(f"image matching (RANSAC)...done", end='\n')

    # image stiching
    print(f"image stiching...", end='\r')
    big_image = imageStiching(images, theta_list)
    print(f"image stiching...done", end='\n')

    im = Image.fromarray(np.uint8(big_image))
    im.save(f"../data/output/big_image.jpg")

