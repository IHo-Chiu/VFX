
import numpy as np

def cylindricalProjection(image, focal_length):
    height , width , channel = image.shape
    h = height/2 / np.sqrt((width/2) ** 2 + focal_length ** 2)
    h = int(h * focal_length)
    theta = np.arctan(width/2 / focal_length)
    theta = int(theta * focal_length)
    
    # print("old size = ", (height, width))
    # print("new size = ", (h*2, theta*2))
    
    # x' = s * arctan(x/f) => tan(x'/s) * f = x
    # y' = s * y/(sqrt(x^2 + f^2)) => y'/s * sqrt(x^2 + f^2) = y
    result = np.zeros((h*2, theta*2, channel))
    for x in range(-theta, theta):
        xx = np.tan(x / focal_length) * focal_length
        a = np.sqrt(xx ** 2 + focal_length ** 2) / focal_length
        for y in range(-h, h):
            yy = y * a
            result[y+h, x+theta] = image[int(yy+height/2), int(xx+width/2)]
    
    return result