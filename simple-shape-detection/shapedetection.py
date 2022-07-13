import numpy as np
import cv2

cap = cv2.VideoCapture(0)
# img = cv2.imread('live.png')

while True:
    ret, frame = cap.read()
    # file_name = cv2.imwrite('frame.png', frame)
    # img = cv2.imread(frame)
    
    img = cv2.resize(frame, None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)

    # get a blank canvas for drawing contour on and convert img to grayscale
    canvas = np.zeros(img.shape, np.uint8)
    img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # filter out small lines between counties
    kernel = np.ones((5,5),np.float32)/25
    img2gray = cv2.filter2D(img2gray,-1,kernel)

    # threshold the image and extract contours
    ret,thresh = cv2.threshold(img2gray,250,255,cv2.THRESH_BINARY_INV)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    # find the main island (biggest area)
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)

    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)

    # define main island contour approx. and hull
    perimeter = cv2.arcLength(cnt,True)
    epsilon = 0.01*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)

    hull = cv2.convexHull(cnt)

    # cv2.isContourConvex(cnt)

    cv2.drawContours(img, cnt, -1, (0, 255, 0), 3)
    cv2.drawContours(img, approx, -1, (0, 0, 255), 3)
    ## cv2.drawContours(canvas, hull, -1, (0, 0, 255), 3) # only displays a few points as well.

    cv2.imshow("Contour", img)
    k = cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()