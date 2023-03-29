
import numpy as np
from PIL import Image
import argparse
import cv2

def reinhard_tone_mapping_algorithm(hdr_image, alpha=0.8):

    # Step1. get luminance
    h, w, c = hdr_image.shape
    R_w = hdr_image[ :, :, 0]
    G_w = hdr_image[ :, :, 1]
    B_w = hdr_image[ :, :, 2]
    L_w = np.zeros((h, w), dtype=float)
    L_w = R_w * 0.2126 + G_w * 0.7152 + B_w * 0.0722
    
    # Step2. calculate average luminance
    L_avg = np.exp(np.mean(np.log(L_w + 1e-8)))

    # Step3. adjust new luminance with hyperparameter a
    L_m = L_w * alpha / L_avg

    # Step4. map luminance to [0, 1]
    L_d = L_m * (L_m / np.max(L_m) ** 2 + 1) / (L_m + 1)
    L_d = L_m / (L_m + 1)

    # Step5. reconstruct RGB
    R_d = R_w * L_d / L_w
    G_d = G_w * L_d / L_w
    B_d = B_w * L_d / L_w

    ldr_image = np.zeros_like(hdr_image)
    ldr_image[ :, :, 0] = R_d
    ldr_image[ :, :, 1] = G_d
    ldr_image[ :, :, 2] = B_d

    # map to [0, 255]
    ldr_image = np.clip(255 * ldr_image , 0 , 255).astype(np.uint8)

    return ldr_image

def main(args):
    hdr_image = np.load(args.npy_path)

    ldr_image = reinhard_tone_mapping_algorithm(hdr_image, alpha=0.8)
    
    hdr_image = np.clip(255 * hdr_image , 0 , 255).astype(np.uint8)
    
    img = Image.fromarray(np.uint8(hdr_image))
    img.save('data/hdr_image.png')

    img = Image.fromarray(np.uint8(ldr_image))
    img.save('data/tone_mapping_image.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reinhard Tone Mapping Algorithm')
    parser.add_argument('npy_path', help='HDR Numpy image')
    args = parser.parse_args()
    main(args)