import cv2
import numpy as np

img = cv2.imread('basictest.png')
height, width = img.shape[:2]

start_height = int(height/2)
end_hight = height
pos_x = int(width/2)
colors = []
for pos_y in range(start_height, end_hight):
    color = img[pos_y, pos_x]
    colors.append(color)

print(colors)