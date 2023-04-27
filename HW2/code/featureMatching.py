
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def keypoint_matching(des1, kps1, des2, kps2, threshold = 0.9):
    matched_kp = []
    cosine_matrix = cosine_similarity(des1, des2)
    for i in range(len(cosine_matrix)):
        max_index = np.argmax(cosine_matrix[i])
        max_match = cosine_matrix[i][max_index]

        cosine_matrix[i][max_index] = 0
        second_index = np.argmax(cosine_matrix[i])
        second_match = cosine_matrix[i][second_index]
        if max_match > 0 and second_match/max_match < threshold:
            matched_kp.append([kps1[i], kps2[max_index]])

    return matched_kp
