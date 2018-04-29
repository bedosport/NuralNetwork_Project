import cv2
from collections import Counter
import numpy as np


def segment(orig_img, colored):
    objectsList = []
    objectsPos = []
    img = cv2.imread(colored)  # colored image
    height, width, channels = img.shape
    img2 = cv2.imread(orig_img)  # original image

    x = Counter([tuple(colors) for i in img for colors in i])
    # x counter to count the most frequently colors in the img
    commons = x.most_common(5)

    max_area = -1
    max_segm_ind = 0
    counter = 0
    segments = [1, 1, 1, 1, 1]
    x = range(20)
    w, h = 4, 5
    dimensions = [[0 for x in range(w)] for y in range (h)]
    # to Exclude tha largest area ( always the background has this feature) O(5*H*W)
    for k, count in commons:
        xmin = 10000
        ymin = 10000
        xmax = -1
        ymax = -1
        r_c = k[0]
        g_c = k[1]
        b_c = k[2]

        for i in range(height):
            for j in range(width):
                r = img[i, j, 0]
                g = img[i, j, 1]
                b = img[i, j, 2]
                if (r == r_c) and (g == g_c) and (b == b_c):
                    xmin = min(xmin, j)
                    xmax = max(xmax, j)
                    ymin = min(ymin, i)
                    ymax = max(ymax, i)
        area = (ymax - ymin + 1) * (xmax - xmin + 1)
        dimensions[counter][0] = xmin
        dimensions[counter][1] = xmax
        dimensions[counter][2] = ymin
        dimensions[counter][3] = ymax

        if area > max_area:
            max_area = area
            max_segm_ind = counter
        counter = counter + 1

    segments[max_segm_ind] = 0
    f = 1  # old method to skip the background not always true ( components has largest # of pixels)
    counter = 0

    for c , count in commons:  # O(5*[h*w]) h,w for each object

        x = segments[counter]
        if x == 0:
            counter = counter + 1
            continue

        r_c = c[0]
        g_c = c[1]
        b_c = c[2]
        xmin = dimensions[counter][0]
        xmax = dimensions[counter][1]
        ymin = dimensions[counter][2]
        ymax = dimensions[counter][3]

        area = (ymax - ymin + 1) * (xmax - xmin + 1)
        cnt = 0
        for ii in range(ymin, ymax):
            for jj in range(xmin, xmax):
                r_t = img[ii, jj, 0]
                g_t = img[ii, jj, 1]
                b_t = img[ii, jj, 2]
                if (r_t == r_c) and (g_t == g_c) and (b_t == b_c):
                    cnt = cnt + 1

        ratio = float(cnt / area)
        counter = counter + 1

        if ratio > 0.2:
            crop_img = img2[ymin:ymin + ymax - ymin + 1, xmin:xmin + xmax - xmin + 1]  # Crop img[y: y + h, x: x + w]
            #crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            #crop_img = cv2.resize(crop_img, (50, 50))
            #crop_img = np.reshape(crop_img, 2500)
            objectsList.append(crop_img)
            objectsPos.append((xmin, xmax, ymin, ymax))
    return objectsList, objectsPos
