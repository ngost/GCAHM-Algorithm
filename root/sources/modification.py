import cv2
import numpy as np
from matplotlib import pyplot as plt


def adjust_gamma(image, gamma=0.5):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((j / 255.0) ** invGamma) * 255
                      for j in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# first read image and origin plt show


img = cv2.imread('airplan.png', cv2.IMREAD_UNCHANGED)
hist, bins = np.histogram(img.flatten(), 256, [0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

# calc histogram average value
i = 0
avg = 0
avg_cum = 0
for avg in hist:
    avg_cum = avg + avg_cum
    i = i+1
    # print(avg_cum)
# print(avg_cum/256)
avg_cum = avg_cum/256

# calc cum2
cum2 = 0
avg = 0
for avg in hist:
    re = ((avg - avg_cum)**2)
    cum2 = cum2 + re

cum2 = (cum2/256)**0.5
cum2 = int(cum2)
# addition into origin histogram

avg = 0
hist_zero = np.full(256, cum2)
hist_m = hist + hist_zero
hist = hist_m
# end histo modifying------------------------------------

img = adjust_gamma(img,0.8)
# write non-gammaCorrection

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()
# modified histogram show

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
# equalizationed

img2 = cdf[img]
hist, bins = np.histogram(img2.flatten(), 256, [0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()
plt.plot(cdf_normalized, color='b')
plt.hist(img2.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.show()
cv2.imwrite("test_gcahm.jpeg",img2)
# file out