import timeit
from collections import Counter, defaultdict
from disjoint_set import DisjointSet
import numpy as np
import cv2
import matplotlib.pyplot as plt


timing_number = 100  # number of times to time a function
width = 100
height = 100
img=cv2.imread("multiple_objects.png", cv2.IMREAD_GRAYSCALE)
# img=cv2.imread("glue_gray.png", cv2.IMREAD_GRAYSCALE)
# img=cv2.imread("sample.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (width, height))
threshold = 25
# threshold = 40

binary_image = threshold <= img

##

plt.imshow(binary_image, cmap="gray")
plt.show(block = False)

##

def sequentialLabeling(binary_image):
    labels = np.zeros_like(binary_image, dtype="int")
    labels[0][0] = 1
    labels[0][1] = 2
    labels[1][0] = 3
    new_label = 4


    disjoint_set = DisjointSet()

    for y in range(1, binary_image.shape[0]):
        for x in range(1, binary_image.shape[1]):
            if binary_image[y][x] == 0:
                labels[y][x] = 0
            elif binary_image[y-1][x-1] == 0 and binary_image[y-1][x] == 0 and binary_image[y][x-1] == 0:
                labels[y][x] = new_label
                new_label += 1
            elif labels[y-1][x-1] != 0:
                labels[y][x] = labels[y-1][x-1]
            elif labels[y][x-1] != 0 and labels[y-1][x] == 0:
                labels[y][x] = labels[y][x-1]
            elif labels[y-1][x] != 0 and labels[y][x-1] == 0:
                labels[y][x] = labels[y-1][x]
            elif labels[y-1][x] == labels[y][x-1]:
                labels[y][x] = labels[y-1][x]
            else:
                assert labels[y-1][x] != labels[y][x-1]
                disjoint_set.union(labels[y-1][x], labels[y][x-1])
                # equivalence[labels[y-1][x]].append(labels[y][x-1])
                labels[y][x] = labels[y-1][x]

    for y in range(0, binary_image.shape[0]):
        for x in range(0, binary_image.shape[1]):
            if binary_image[y][x] != 0:
                labels[y][x] = disjoint_set.find(labels[y][x])
            else:
                labels[y][x] = 0
    return labels

time_sequentialLabeling = timeit.timeit(stmt="sequentialLabeling(binary_image)", globals=globals(), number=timing_number)
print(f"time_sequentialLabeling = {time_sequentialLabeling}")

labels = sequentialLabeling(binary_image)
unique_colors = len(Counter(labels.flatten()))
print(f"unique_colors = {unique_colors}")
im = plt.imshow(labels)
plt.show(block = True)


##

def disjointSetLabeling(binary_image):
    disjoint_set = DisjointSet()
    disjoint_set._data

    for y in range(0, binary_image.shape[0]):
        for x in range(0, binary_image.shape[1]):
            for i, j in ((-1, -1), (-1, 0), (0, -1)):
                new_y = y + i
                new_x = x + j 
                if new_y >= 0 and new_x >= 0 and binary_image[y][x] and binary_image[new_y][new_x]:
                    disjoint_set.union(new_y * binary_image.shape[1] + new_x, y * binary_image.shape[1] + x)

    labels = np.zeros_like(binary_image, dtype="int")

    for y in range(0, binary_image.shape[0]):
        for x in range(0, binary_image.shape[1]):
            if binary_image[y][x]:
                labels[y][x] = disjoint_set.find(y * binary_image.shape[1] + x)
    
    return labels

time_disjointSetLabeling = timeit.timeit(stmt="disjointSetLabeling(binary_image)", globals=globals(), number=timing_number)
print(f"time_disjointSetLabeling = {time_disjointSetLabeling}")


labels = disjointSetLabeling(binary_image)
unique_colors = len(Counter(labels.flatten()))
print(f"unique_colors = {unique_colors}")
im = plt.imshow(labels)
plt.show(block = True)

