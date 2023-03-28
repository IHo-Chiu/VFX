
import numpy as np
import argparse
from utils import load_images, down_sample

def debevec_HDR_algorithm_1_channel(images, exposures):
    W = np.asarray([i/128 if i <= 128 else (256-i)/128 for i in range(256)], dtype=float)
    Z = down_sample(images, scale=32)
    n = len(images)
    h, w = Z[0].shape
    p = h*w

    A = np.zeros((n*p + 1 + 254, 256 + n), dtype=float)
    b = np.zeros((n*p + 1 + 254), dtype=float)

    for i in range(n):
        for j in range(h):
            for k in range(w):
                z = Z[i][j][k]
                A[i*p + j*w + k][z] = W[z] * 1
                A[i*p + j*w + k][256+i] = W[z] * -1
                b[i*p + j*w + k] = W[z] * np.log(exposures[i])
    
    A[n*p][128] = 1
    b[n*p] = 0

    for i in range(254):
        A[n*p+i+1][i] = W[i+1] * 1
        A[n*p+i+1][i+1] = W[i+1] * -2
        A[n*p+i+1][i+2] = W[i+1] * 1

    x = np.linalg.lstsq(A, b, rcond=None)[0]
    G = x[:256]

    # reconstruct hdr image
    Z = images
    E_up = np.zeros_like(Z[0], dtype=float)
    E_down = np.zeros_like(Z[0], dtype=float)
    for i in range(len(Z)):
        z = Z[i]
        t = exposures[i]
        E_up += (G[z] - np.log(t)) * W[z]
        E_down += W[z]
    
    E = np.exp(E_up/E_down)

    return E


def debevec_HDR_algorithm(images, exposures):

    r_images = []
    g_images = []
    b_images = []

    for image in images:
        r_images.append(image[:, :, 0])
        g_images.append(image[:, :, 1])
        b_images.append(image[:, :, 2])

    hdr_images = np.zeros(images[0].shape)
    hdr_images[ :, :, 0] = debevec_HDR_algorithm_1_channel(r_images, exposures)
    hdr_images[ :, :, 1] = debevec_HDR_algorithm_1_channel(g_images, exposures)
    hdr_images[ :, :, 2] = debevec_HDR_algorithm_1_channel(b_images, exposures)

    return hdr_images

def main(args):
    images, exposures = load_images(args.csv_path)
    hdr_image = debevec_HDR_algorithm(images, exposures)

    with open('data/hdr_image_debevec.npy', 'wb') as f:
        np.save(f, hdr_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Debevec HDR Algorithm')
    parser.add_argument('csv_path', help='Image list with exposure')
    args = parser.parse_args()
    main(args)