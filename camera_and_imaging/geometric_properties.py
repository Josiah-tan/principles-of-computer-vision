import numpy as np
import cv2
import matplotlib.pyplot as plt
# https://www.coursera.org/learn/cameraandimaging/ungradedWidget/Ijloy/4-2-geometric-properties
width = 100
height = 100
img=cv2.imread("spanner_gray.png", cv2.IMREAD_GRAYSCALE)
# img=cv2.imread("glue_gray.png", cv2.IMREAD_GRAYSCALE)
# img=cv2.imread("sample.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (width, height))
# img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
# plt.imshow(img)
# plt.show()

## finding an appropriate threshold for usage

# plt.hist(img.flatten().tolist(), bins=10)
# plt.show()

## thresholding the image

threshold = 25
# threshold = 40

thresholded_image = threshold <= img
# plt.imshow(thresholded_image)
# plt.show()

## zeroth moment calculation

A = np.sum(thresholded_image)  # area

x_bar = sum(sum(x * column) for x, column in enumerate(thresholded_image.T)) / A
print(f"x_bar = {x_bar}")
y_bar = sum(sum(y * row) for y, row in enumerate(thresholded_image)) / A
print(f"y_bar = {y_bar}")

plt.annotate("$\\bar{x}$, $\\bar{y}$", (x_bar, y_bar))
# plt.imshow(img)
# plt.show()


## axis least second moment

a = sum(sum(column * (x - x_bar) ** 2) for x, column in enumerate(thresholded_image.T))
print(f"a = {a}")
b = sum(sum(value * (x - x_bar) * (y - y_bar) for x, value in enumerate(row)) for y, row in enumerate(thresholded_image)) * 2
print(f"b = {b}")
c = sum(sum(row * (y - y_bar) ** 2) for y, row in enumerate(thresholded_image))
print(f"c = {c}")
 
theta_min_degrees = 0.5 * (np.arctan(b / ( a - c))) * 180 / np.pi
theta_max_degrees = theta_min_degrees + 90

theta_min = theta_min_degrees * np.pi / 180
theta_max = theta_max_degrees * np.pi / 180

if (a - c) * np.cos(2 * theta_min) + b * np.sin(2 * theta_min) < 0:  # concave down → max
    theta_max, theta_min = theta_min, theta_max

positive_concavity = (a - c) * np.cos(2 * theta_min) + b * np.sin(2 * theta_min)
assert positive_concavity > 0
negative_concavity = (a - c) * np.cos(2 * theta_max) + b * np.sin(2 * theta_max)
assert negative_concavity < 0

theta_min_degrees = theta_min * 180 / np.pi
print(f"theta_min_degrees = {theta_min_degrees}")
theta_max_degrees = theta_max * 180 / np.pi
print(f"theta_max_degrees = {theta_max_degrees}")

## the lines: x sin(theta) - y cos(theta) + rho = 0

rho_min = - x_bar * np.sin(theta_min) + y_bar * np.cos(theta_min)
print(f"rho_min = {rho_min}")
rho_max = - x_bar * np.sin(theta_max) + y_bar * np.cos(theta_max)
print(f"rho_max = {rho_max}")

x = np.array([i for i in range(width)])
y_max = (x * np.sin(theta_max) + rho_max) / np.cos(theta_max)
y_min = (x * np.sin(theta_min) + rho_min) / np.cos(theta_min)

## roundedness

E_min = a * np.sin(theta_min) ** 2 - b * np.sin(theta_min) * np.cos(theta_min) + c * np.cos(theta_min) ** 2
E_max = a * np.sin(theta_max) ** 2 - b * np.sin(theta_max) * np.cos(theta_max) + c * np.cos(theta_max) ** 2
roundedness = E_min / E_max
print(f"roundedness = {roundedness}")

## plotting 

plt.plot(x, y_max)
plt.plot(x, y_min, label="min")
plt.imshow(thresholded_image, cmap="gray")
plt.legend()
plt.show()
