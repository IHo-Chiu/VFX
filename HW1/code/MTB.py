
import numpy as np
import argparse
from utils import load_images
from PIL import Image

def gray_scale(rgb):
    return rgb[ :, :, 0] * 0.2126 + rgb[ :, :, 1] * 0.7152 + rgb[ :, :, 2] * 0.0722

def mtb_align(im1, im2, search_range=30):
    # Convert the images to grayscale
    im1_gray = gray_scale(im1)
    im2_gray = gray_scale(im2)
 
    # Compute the median intensity of the two images
    im1_median = np.median(im1_gray)
    im2_median = np.median(im2_gray)
 
    # Compute the threshold values for the two images
    im1_thresh = (im1_gray > im1_median).astype(np.uint8) * 255
    im2_thresh = (im2_gray > im2_median).astype(np.uint8) * 255
 
    # Initialize the best match offset and error
    best_offset = [0, 0]
    min_error = np.inf
 
    # Loop over all possible offsets within the search range
    for dx in range(-search_range, search_range + 1):
        for dy in range(-search_range, search_range + 1):
            # Shift the second image by the current offset
            if dx >= 0:
                im2_shifted = im2_thresh[:, dx:]
                im2_shifted = np.pad(im2_shifted, ((0, 0), (0, dx)), mode='constant', constant_values=0)
            else:
                im2_shifted = im2_thresh[:, :dx]
                im2_shifted = np.pad(im2_shifted, ((0, 0), (-dx, 0)), mode='constant', constant_values=0)
 
            if dy >= 0:
                im2_shifted = im2_shifted[dy:, :]
                im2_shifted = np.pad(im2_shifted, ((0, dy), (0, 0)), mode='constant', constant_values=0)
            else:
                im2_shifted = im2_shifted[:dy, :]
                im2_shifted = np.pad(im2_shifted, ((-dy, 0), (0, 0)), mode='constant', constant_values=0)
 
            # Compute the error between the two thresholded images
            error = np.sum(np.abs(im1_thresh - im2_shifted))
 
            # Update the best match offset and error if necessary
            if error < min_error:
                best_offset = [dx, dy]
                min_error = error
 
    # Return the best match offset
    return best_offset

def shift_image(X, dx, dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy>0:
        X[:dy, :] = 0
    elif dy<0:
        X[dy:, :] = 0
    if dx>0:
        X[:, :dx] = 0
    elif dx<0:
        X[:, dx:] = 0
    return X

def mtb(images):
    template = images[len(images)//2]
    shifted_images = []
    for image in images:
        shift = mtb_align(template, image, search_range=5)
        shifted_image = shift_image(image, -shift[0], -shift[1])
        shifted_images.append(shifted_image)

    return shifted_images


def main(args):
    images, exposures, paths = load_images(args.csv_path, return_images_name = True)
    shifted_images = mtb(images)

    for i, images in enumerate(shifted_images):
        img = Image.fromarray(np.uint8(images))
        img.save(f'data/shifted_{paths[i]}')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robertson HDR Algorithm')
    parser.add_argument('csv_path', help='Image list with exposure')
    args = parser.parse_args()
    main(args)