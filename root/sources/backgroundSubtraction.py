import numpy as np
import cv2

img = cv2.imread("couple.jpg")

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
fgbg = cv2.createBackgroundSubtractorMOG2()

result = fgbg.apply(img)
cv2.imwrite("couple_backgroundSubtraction.jpg", result)
