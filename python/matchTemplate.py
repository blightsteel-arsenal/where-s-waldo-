import cv2
import numpy as np
import math
import time
import csv

numImgs = 7
GT = [[64, 511], [1383, 454], [1467, 265], [1709, 406], [221, 725], [808, 950], [1152, 287]]
orgPath = "./datasets/originals/"
templatePath = "./datasets/templates/"
imgExt = ".jpg"
similarityFunctions = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

def euclidean_distance(pt1, pt2):    
    pt1x = int(pt1[0])
    pt1y = int(pt1[1])
    pt2x = int(pt2[0])
    pt2y = int(pt2[1])
    xDifferenceSquared = (pt1x - pt2x) ** 2
    yDifferenceSquared = (pt1y - pt2y) ** 2
    return  math.sqrt(xDifferenceSquared + yDifferenceSquared)

if __name__ == "__main__":

    time_file = open('elapsed_time.csv', 'w', newline='')
    error_file = open('error.csv', 'w', newline='')
    time_writer = csv.writer(time_file)
    error_writer = csv.writer(error_file)

    first_row = ["function","img1","img2","img3","img4","img5","img6","img7"]

    time_writer.writerow(first_row)
    error_writer.writerow(first_row)

    for functionStr in similarityFunctions:
        timeRow = [functionStr]
        errorRow = [functionStr]

        for i in range(1,numImgs+1):

            orgImgPath = orgPath+str(i)+imgExt
            tempImgPath = templatePath+str(i)+imgExt

            orgImg = cv2.imread(orgImgPath, cv2.IMREAD_COLOR)
            img = orgImg.copy()
            template = cv2.imread(tempImgPath, cv2.IMREAD_COLOR)

            start = time.time()
            function = eval(functionStr)
            res = cv2.matchTemplate(img,template,function)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if function in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc

            end = time.time()

            errorRow.append(str(euclidean_distance(top_left, GT[i-1])))
            timeRow.append(str(end-start))

        time_writer.writerow(timeRow)
        error_writer.writerow(errorRow)

    error_file.close()
    time_file.close()