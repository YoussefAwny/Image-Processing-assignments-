import cv2
import matplotlib
matplotlib.use('Agg')
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np


def insta_like(image,filter):
    if filter == "gotham":
        return gotham(image)
    elif filter == "black and white":
        return blacknwhite(image)
    elif filter == "california":
        return california(image)
    elif filter == "lord kelvin":
        return kelvin(image)
    elif filter == "excited":
        return excited(image)
    else:
        print("Please enter filter type")


def gotham(img):
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    b = cv2.subtract(b, 35)
    g = cv2.subtract(g, 40)
    r = cv2.subtract(r, 20)
    op = cv2.merge((b, g, r))
    return op


def blacknwhite(img):
    op = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    op = cv2.subtract(op,35)
    return op


def california(img):
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    b = cv2.add(b, 46)
    g = cv2.subtract(g, 11)
    r = cv2.subtract(r, 21)

    op = cv2.merge((b, g, r))
    kernel = np.ones((2, 2), 'uint8')
    op = cv2.erode(op, kernel, iterations=1)
    return op


def kelvin(img):
    op = img
    layer = np.zeros(img.shape, np.uint8)

    layer[:, :, 0] = 68
    layer[:, :, 1] = 84
    layer[:, :, 2] = 222

    b = original_image[:, :, 0]
    g = original_image[:, :, 1]
    r = original_image[:, :, 2]

    # b = cv2.add(b, 50)
    # g = cv2.subtract(g, 50)
    # r = cv2.subtract(r, 21)

    img = cv2.merge((b,g,r))

    cv2.addWeighted(img,0.3,layer,0.7,0,op)
    return op


def excited(img):
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    b = cv2.add(b, 0)
    #ret, b = cv2.threshold(b,40,1,cv2.THRESH_BINARY)
    ret, g = cv2.threshold(g,100,255,cv2.THRESH_BINARY)
    r = cv2.add(r, 255)


    op = cv2.merge((b, g, r))
    return op



original_image = cv2.imread("filter.jpg")
cv2.imshow("original", original_image)


# gotham = gotham(original_image)   # filter 1 (GOTHAM)
# # --------------------------------------------------------------------------------------------------------------------
# bw = blacknwhite(original_image)  # filter 2 (black and white)
# # --------------------------------------------------------------------------------------------------------------------
# cal = california(original_image)  # filter 3 (California)
# # --------------------------------------------------------------------------------------------------------------------
# kel = kelvin(original_image)  # filter 4 lord kelvin
# # --------------------------------------------------------------------------------------------------------------------
# ex = excited(original_image)  # filter 5 excited

go = insta_like(original_image, "gotham")
bw = insta_like(original_image, "black and white")
cal = insta_like(original_image, "california")
kel = insta_like(original_image, "lord kelvin")
ex = insta_like(original_image, "excited")

cv2.imshow("go", go)
cv2.imshow("bw", bw)
cv2.imshow("cal",cal)
cv2.imshow("kel", kel)

cv2.imshow("ex", ex)


cv2.waitKey()