
import math
import random
import numpy as np

def energy_func(theta, pts_pair_list):
    sum = 0
    n = len(pts_pair_list)
    for i in range(n):
        pt1 = pts_pair_list[i][0]
        pt2 = pts_pair_list[i][1]
        sum += (theta[0] + pt1[0] - pt2[0]) ** 2 + (theta[1] + pt1[1] - pt2[1]) ** 2

    return sum/n

def calc_theta(pts_pair_list):
    m1 = 0
    m2 = 0
    n = len(pts_pair_list)
    for i in range(n):
        pt1 = pts_pair_list[i][0]
        pt2 = pts_pair_list[i][1]
        m1 += pt2[0] - pt1[0]
        m2 += pt2[1] - pt1[1]

    return m1/n, m2/n

def RANSAC(pts_pair_list, p=0.2, P=0.99, n=6):
    k = int(math.log(1-P) / math.log(1-p**n))
    best_m1 = 0
    best_m2 = 0
    best_loss = 1000000
    for i in range(k):
        sample_pts_pair_list = random.sample(pts_pair_list, n)
        theta = calc_theta(sample_pts_pair_list)
        loss = energy_func(theta, sample_pts_pair_list)
        if loss < best_loss:
            best_loss = loss
            best_m1 = theta[0]
            best_m2 = theta[1]

    return np.array([best_m1, best_m2])