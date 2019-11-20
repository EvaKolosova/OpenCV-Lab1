import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

## --- task1 - оригинальное изображение
imageSource = '/home/ekolosova/Desktop/bear.jpeg'
image = cv.imread(imageSource)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

if image is not None:
    cv.imshow('Original image', image) ## вывод исходного изображения


    ## --- task2 - полутоновое изображение
    cv.imshow('Gray sample', gray)  ## вывод полутонового изображения
elif image is None:
    print("Error loading image")


## --- task3 - улучшенная контрастность
clahe = cv.createCLAHE(clipLimit=3., tileGridSize=(8, 8)) # CLAHE (Contrast Limited Adaptive Histogram Equalization)

lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)  # convert from BGR to LAB color space
l, a, b = cv.split(lab)  # split on 3 different channels

l2 = clahe.apply(l)  # apply CLAHE to the L-channel

lab = cv.merge((l2, a, b))  # merge channels
image2 = cv.cvtColor(lab, cv.COLOR_LAB2BGR)  # convert from LAB to BGR
cv.imshow('Increased contrast', image2) ## изображение с улучшеным контрастом


## --- task4 - Canny - края обьектов
edges = cv.Canny(image, 80, 150)
cv.imshow('Edges from Canny', edges) ## изображение с Canny - края обьектов


## --- task5 - угловые точки обьектов, нарисовать кругом с радиусом=2 в изображение с краями
source_window = 'Source image'
corners_window = 'Corners detected'
max_thresh = 255

def cornerHarris_demo(val):
    thresh = val

    # Detector parameters
    blockSize = 2
    apertureSize = 3
    k = 0.04

    # Detecting corners
    dst = cv.cornerHarris(gray, blockSize, apertureSize, k)

    # Normalizing
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    dst_norm_scaled = cv.convertScaleAbs(dst_norm)

    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i,j]) > thresh:
                cv.circle(dst_norm_scaled, (j,i), 2, (0), 2) #радиус=2

    # Showing the result
    cv.namedWindow(corners_window)
    cv.imshow(corners_window, dst_norm_scaled)

# Create a window and a trackbar
cv.namedWindow(source_window)
thresh = 200 # initial threshold
cv.createTrackbar('Threshold: ', source_window, thresh, max_thresh, cornerHarris_demo)
cv.imshow(source_window, image)
cornerHarris_demo(thresh)


## --- task6 - для границ и угловых точек построить карту расстояний
data = np.array(gray)
ret, binary = cv.threshold(data, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU) #пороговое=бинарное изображение
cv.imshow('Binary image', binary)

# Perform the distance transform algorithm
dist = cv.distanceTransform(binary, cv.DIST_L2, 3)
# Normalize the distance image for range = {0.0, 1.0}
# so we can visualize and threshold it
cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
cv.imshow('Distance Transform Image', dist)


## --- task7 - в каждом пикселе фильтрация усреднением
kernel = np.ones((5, 5), np.float32)/25
dst = cv.filter2D(image, -1, kernel)

plt.subplot(121), plt.imshow(image), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()


## --- task8 - интегральные изображения в фильтрации усреднением



k = cv.waitKey(0)
cv.destroyAllWindows()






