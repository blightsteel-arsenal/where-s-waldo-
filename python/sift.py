import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

orgPath = "./datasets/clipped/"
templatePath = "./datasets/templates/"
imgExt = ".jpg"

if __name__ == "__main__":
    img1 = cv2.imread(orgPath+"2"+imgExt, 0)
    img2 = cv2.imread('./datasets/waldo.png', 0)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()