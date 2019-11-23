import cv2 as cv
import numpy as np


def cornerHarris_demo(thresh):
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
    print("corners", dst_norm_scaled)

    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i, j]) > thresh:
                cv.circle(dst_norm_scaled, (j, i), 2, 0, 2)  # радиус=2

    # Showing the result
    cv.namedWindow(corners_window)
    cv.imshow(corners_window, dst_norm_scaled)


def limit(value, min, max):
    if value < min:
        return min
    if value > max:
        return max
    return value


def average(data, dist, clear_image, k=150):
    res_image = data.copy()

    for x in range(data.shape[0]):
        for y in range(data.shape[1]):

            step = int(k * dist[x][y] / 2)

            x_new = limit(x + step + 1, 0, data.shape[0] - 1)
            y_new = limit(y + step + 1, 0, data.shape[1] - 1)
            x_old = limit(x - step, 0, data.shape[0] - 1)
            y_old = limit(y - step, 0, data.shape[0] - 1)

            integral_formula = clear_image[x_new, y_new] + clear_image[x_old, y_old] - clear_image[x_new, y_old] - \
                             clear_image[x_old, y_new]  # формула точек в интегральном изображении

            if(step != 0):
                integral_formula /= (x_new - x_old) * (y_new - y_old)
                res_image[x][y] = integral_formula

    return res_image

## --- оригинальное изображение
imageSource = '/home/ekolosova/Desktop/bear.jpg'
image = cv.imread(imageSource)

if image is not None:
    cv.imshow('Original image', image)  ## вывод исходного изображения

    ## --- полутоновое изображение
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray sample', gray)  ## вывод полутонового изображения
elif image is None:
    print("Error loading image")

## --- улучшенная контрастность
clahe = cv.createCLAHE(clipLimit=3., tileGridSize=(8, 8))  # CLAHE (Contrast Limited Adaptive Histogram Equalization)

lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)  # convert from BGR to LAB color space
l, a, b = cv.split(lab)  # split on 3 different channels

l2 = clahe.apply(l)  # apply CLAHE to the L-channel

lab = cv.merge((l2, a, b))  # merge channels
image2 = cv.cvtColor(lab, cv.COLOR_LAB2BGR)  # convert from LAB to BGR
cv.imshow('Increased contrast', image2)  ## изображение с улучшеным контрастом

## --- Canny - края обьектов
edges = cv.Canny(image, 80, 150)
print("edges", edges)
cv.imshow('Edges from Canny', edges)  ## изображение с Canny - края обьектов
edges_temp = cv.bitwise_not(edges)

## --- угловые точки обьектов, нарисовать кругом с радиусом=2 в изображение с краями
source_window = 'Source image'
corners_window = 'Corners detected'
max_thresh = 255

# Create a window and a trackbar
cv.namedWindow(source_window)
thresh = 200  # initial threshold
cv.createTrackbar('Threshold: ', source_window, thresh, max_thresh, cornerHarris_demo)
cv.imshow(source_window, image)
cornerHarris_demo(thresh)

## --- для границ и угловых точек построить карту расстояний
# data = np.array(gray)
# ret, binary = cv.threshold(data, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU) #пороговое=бинарное изображение
# cv.imshow('Binary image', binary)

dist = cv.distanceTransform(edges_temp, cv.DIST_L2, 3)
dist_not_norm = dist
print("dist_transform", dist_not_norm)
cv.normalize(dist_not_norm, dist_not_norm, 0, 1.0, cv.NORM_MINMAX)
cv.imshow('Distance Transform Image', dist_not_norm)

## --- интегральные изображения в фильтрации усреднением
integral_image = cv.integral(gray)
blur_image_int = average(gray, dist, integral_image)

cv.imshow("Blur with integral image", cv.UMat(blur_image_int))


k = cv.waitKey(0)
cv.destroyAllWindows()
