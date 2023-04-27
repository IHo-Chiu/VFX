
import numpy as np
from PIL import Image

def imageStiching(images, theta_list):
    # stiching
    big_image = np.zeros((len(images), 1700, 12000, 3))
    h, w, c = images[0].shape
    original_offset_x = 100
    original_offset_y = 9000

    offset_x = original_offset_x
    offset_y = original_offset_y
    max_offset_x = offset_x
    min_offset_x = offset_x
    big_image[0, offset_x:h+offset_x, offset_y:w+offset_y] = images[0][0:h, 0:w]
    for i, theta in enumerate(theta_list[:-1]):
        offset_x -= int(theta[1])
        offset_y -= int(theta[0])
        min_offset_x = min(min_offset_x, offset_x)
        max_offset_x = max(max_offset_x, offset_x)
        big_image[i+1, offset_x:h+offset_x, offset_y:w+offset_y] = images[i+1][0:h, 0:w]

    big_image = np.ma.median(np.ma.masked_equal(big_image,0),axis=0).data

    # im = Image.fromarray(np.uint8(big_image))
    # im.save(f"after_stiching.jpg")

    # (bonus) Fix up the end-to-end alignment
    end_to_end_alignment_image = np.zeros_like(big_image)

    final_offset_x = original_offset_x
    final_offset_y = original_offset_y
    for i, theta in enumerate(theta_list):
        final_offset_x -= int(theta[1])
        final_offset_y -= int(theta[0])

    w_total = original_offset_y + w - final_offset_y
    offset_x_diff = final_offset_x - original_offset_x
    max_x = max_offset_x+offset_x_diff
    min_x = min_offset_x+h-offset_x_diff
    w_count = 0
    for j in range(offset_y, original_offset_y+w):
        w_count += 1
        for i in range(max_offset_x+offset_x_diff, min_offset_x+h-offset_x_diff):
            offset = offset_x_diff * w_count / w_total
            if big_image[i-int(offset), j].sum() != 0:
                max_x = max(max_x, i)
                min_x = min(min_x, i)
                end_to_end_alignment_image[i, j] = big_image[i-int(offset), j]
    big_image = end_to_end_alignment_image

    # im = Image.fromarray(np.uint8(big_image))
    # im.save(f"end-to-end.jpg")

    # crop
    big_image = big_image[max_x-h:min_x+h, offset_y:original_offset_y+w]
    return big_image