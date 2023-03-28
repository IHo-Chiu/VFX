
import numpy as np
from PIL import Image
import argparse

def heinhard_tone_mapping_algorithm(hdr_image, alpha=1, beta=1):

    # Step1. get luminance
    h, w, c = hdr_image.shape
    R_w = hdr_image[ :, :, 0]
    G_w = hdr_image[ :, :, 1]
    B_w = hdr_image[ :, :, 2]
    L_w = np.zeros((h, w), dtype=float)
    # L_w = R_w * 0.2126 + G_w * 0.7152 + B_w * 0.0722
    L_w = R_w * 0.299 + G_w * 0.587 + B_w * 0.114
    
    # Step2. calculate average luminance
    L_avg = np.exp(np.sum(np.log(L_w))/(h*w))

    # Step3. adjust new luminance with hyperparameter a
    L_m = L_w * alpha / L_avg

    # Step4. map luminance to [0, 1]
    L_d = L_m * (L_m / beta + 1) / (L_m + 1)

    # Step5. reconstruct RGB
    R_d = R_w * L_d / L_w
    G_d = G_w * L_d / L_w
    B_d = B_w * L_d / L_w

    ldr_image = np.zeros_like(hdr_image)
    ldr_image[ :, :, 0] = R_d
    ldr_image[ :, :, 1] = G_d
    ldr_image[ :, :, 2] = B_d

    # map to [0, 255]
    ldr_image = ldr_image / np.max(ldr_image) * 256
    ldr_image = ldr_image.astype(int)

    return ldr_image

def Reinhard_global(HDR_image , a = 0.18 , L_white = None):
    L_w = 0.06 * HDR_image[ : , : , 2] + 0.67 * HDR_image[ : , : , 1] + 0.27 * HDR_image[ : , : , 0]

    L_w_average = np.exp(np.mean(np.log(L_w + 1e-8)))
    L_m = a * L_w / L_w_average
    if L_white is None:
        L_white = np.max(L_m)
    L_d = L_m * (1 + L_m / L_white**2) / (1 + L_m)
    
    LDR_image = np.zeros(HDR_image.shape)
    for i in range(3):
        LDR_image[ : , : , i] = HDR_image[ : , : , i] / L_w * L_d

    LDR_image = np.clip(255 * LDR_image , 0 , 255).astype(np.uint8)
    return LDR_image

def main(args):
    hdr_image = np.load(args.npy_path)
    ldr_image = Reinhard_global(hdr_image)
    # ldr_image = heinhard_tone_mapping_algorithm(hdr_image, alpha=0.8, beta=2)

    hdr_image = hdr_image * 256 / np.max(hdr_image)
    hdr_image = hdr_image.astype(int)
    
    img = Image.fromarray(np.uint8(hdr_image))
    img.save('data/hdr_image.png')

    img = Image.fromarray(np.uint8(ldr_image))
    img.save('data/tone_mapping_image.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reinhard Tone Mapping Algorithm')
    parser.add_argument('npy_path', help='HDR Numpy image')
    args = parser.parse_args()
    main(args)