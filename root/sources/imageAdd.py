import cv2
import numpy as np
from matplotlib import pyplot as plt


img1 = cv2.imread("f_f_gcahm.jpeg")
img2 = cv2.imread("f_b_gcahm.jpeg")

result = cv2.add(img1,img2,)
cv2.imwrite("f_add_gcahm2.jpeg",result)

