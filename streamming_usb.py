import cv2
import numpy as np
from goprocam import GoProCamera
from goprocam import constants
from PIL import Image
import pytesseract

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    cv2.imshow('Live Video', frame)
 
    height, width = frame.shape[:2]
    # OCR_text = pytesseract.image_to_string(Image.open(frame))
    # if (OCR_text):
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Low and high color range of Mask for extracting square
    low = np.array([0, 42, 0])
    high = np.array([179, 255, 255])
    mask = cv2.inRange(hsv_frame, low, high)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Setting points to scan for detecting color.
    pos_x = int(width/2) 
    start_y = int(height/2)
    end_y = height
    colors = []
    last_color = []

    # Getting color value in points and keeping in array.
    for pos_y in range(start_y, end_y):
        color = result[pos_y, pos_x]
        colors.append(color)

    # Finding out the correct point at non-zero value.
    for out_color in colors:
        if out_color[0] != 0 and out_color[1] != 0 and out_color[2] != 0:
            last_color.append(out_color)

    # Result of color 
    if (len(last_color)/2) != 0:
        for i in range (10):
            position = int(len(last_color)/2) 
            print(last_color[position])
 
    c = cv2.waitKey(5)
    if c == 27:
        break

cv2.destroyAllWindows()
# cap.release()
# cv2.destroyAllWindows()