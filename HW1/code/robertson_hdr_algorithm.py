
import numpy as np
import argparse
from utils import load_images, down_sample
from MTB import mtb

def robertson_HDR_algorithm_1_channel(images, exposures):
    W = (np.exp(4) / (np.exp(4) - 1)) * np.exp(-(4 * np.arange(256) / 255 - 2)**2) + (1 / (1 - np.exp(4)))
    G = np.asarray([i/256 for i in range(256)], dtype=float)
    Z = down_sample(images, scale=8)
    E = np.zeros_like(Z[0], dtype=float)

    loss = 100
    pre_loss = 120
    while abs(loss - pre_loss) > 1:

        # Step1. assume G is known, find E
        E_up = np.zeros_like(E)
        E_down = np.zeros_like(E)
        for i in range(len(Z)):
            z = Z[i]
            t = exposures[i]
            E_up += W[z] * G[z] * t
            E_down += W[z] * t * t
        
        E = E_up/E_down

        # Step2. assume E is known, find G
        G = np.zeros_like(G)
        for i in range(len(Z)):
            z = Z[i]
            t = exposures[i]
            for j in range(256):
                mask = z == j
                G[j] += np.sum((E * t)[mask])

        _, counts = np.unique(Z, return_counts=True)
        G = G / counts

        # Step3. normalize G[128] = 1
        G = G / G[128]

        # Step4. calculate loss
        pre_loss = loss
        loss = 0.0
        for i in range(len(Z)):
            z = Z[i]
            t = exposures[i]
            loss += np.sum(((G[z] - E * t) ** 2) * W[z])

        print(f'loss = {loss}')

    # reconstruct hdr image
    Z = images
    E_up = np.zeros_like(Z[0], dtype=float)
    E_down = np.zeros_like(Z[0], dtype=float)
    for i in range(len(Z)):
        z = Z[i]
        t = exposures[i]
        E_up += W[z] * G[z] * t
        E_down += W[z] * t * t
    
    E = E_up/E_down

    return E

def robertson_HDR_algorithm(images, exposures):

    r_images = []
    g_images = []
    b_images = []

    for image in images:
        r_images.append(image[:, :, 0])
        g_images.append(image[:, :, 1])
        b_images.append(image[:, :, 2])

    hdr_images = np.zeros(images[0].shape)
    hdr_images[ :, :, 0] = robertson_HDR_algorithm_1_channel(r_images, exposures)
    hdr_images[ :, :, 1] = robertson_HDR_algorithm_1_channel(g_images, exposures)
    hdr_images[ :, :, 2] = robertson_HDR_algorithm_1_channel(b_images, exposures)

    return hdr_images

def main(args):
    images, exposures = load_images(args.csv_path, shifted = args.shifted)
    if args.shifted == False:
        images = mtb(images)
    hdr_image = robertson_HDR_algorithm(images, exposures)

    with open('data/hdr_image_robertson.npy', 'wb') as f:
        np.save(f, hdr_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robertson HDR Algorithm')
    parser.add_argument('csv_path', help='Image list with exposure')
    parser.add_argument('--shifted', type=bool, default=False, help='Shifted ot not.')
    args = parser.parse_args()
    main(args)