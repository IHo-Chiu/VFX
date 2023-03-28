

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse

def main(args):
    hdr_image = np.load(args.hdr_npy)

    fig , axes = plt.subplots(1 , 3 , figsize = (16 , 4))
    for i , channel in enumerate(['Red' , 'Green' , 'Blue']):
        ax = axes[i]
        ax.set_title(f'Radiance Map ({channel})')
        im = ax.imshow(np.log(hdr_image[ : , : , i]) , cmap = 'jet')
        ax.set_axis_off()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right' , size = '5%' , pad = 0.1)
        fig.colorbar(im , cax = cax , orientation = 'vertical')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show HDR result.')
    parser.add_argument('hdr_npy', help='Numpy data of HDR image.')
    args = parser.parse_args()
    main(args)