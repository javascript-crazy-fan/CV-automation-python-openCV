import cv2
from PIL import Image
import numpy as np
import pytesseract
import time
import re
import json
import codecs
import os, errno
import imutils
import xlwt


### go pro camera parameters #############
default_altitude, image_pixel, cm_per_pixel = 50*30.48, 382, 0.94

####### reading image from video/camera or file  #######

# file_name = 'ground1.png' # from image file
# cap = cv2.VideoCapture(0) # from camera

cap= cv2.VideoCapture('video.mov') # from video file
########################################################
#### create folders #########
try:
    os.makedirs('binaryimg')
    os.makedirs('boundingimg')
    os.makedirs('cropimg')
    os.makedirs('frameimg')
    os.makedirs('history')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
##################################

### for writing distance data to excel file ###
wb = xlwt.Workbook()
worksheet = wb.add_sheet("Sheet 1", cell_overwrite_ok=True)
distance_list = []
##############################

i = 1 # frame loop variable

while True:
    
    ### For camera/video ###
    ret, img = cap.read()
    if ret == False:
        break
    
    camera_file_name = './frameimg/frameimage.png'
    
    cv2.imwrite(camera_file_name, img)
    img = cv2.imread(camera_file_name)
    cropping_img = Image.open(camera_file_name)
    ##########################################
    
    #-------for image----------------------
    # img = cv2.imread(file_name)
    # cropping_img = Image.open(file_name)
    ########################################

    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    gray = cv2.GaussianBlur(img2gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
            
    ##### for color detection #####
    
    height, width = img.shape[:2] # getting height and width of an image in pixel unit
    hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # convert BGR color to HSV color space

    # Low and high color range of Mask for extracting square
    low = np.array([0, 42, 0])
    high = np.array([179, 255, 255])
    mask = cv2.inRange(hsv_frame, low, high)
    color_img = cv2.bitwise_and(img, img, mask=mask)


    for contour in contours:
        
        [x, y, w, h] = cv2.boundingRect(contour)
        
        if w < 200 or h < 200 or w < 0.9 * h or h < 0.9 * w or w > 0.5*width or h > 0.5*height:
            continue

        delta_text = 15
        delta_color = 50
        
        cv2.rectangle(img, (x + delta_text, y + delta_text), (x + w - delta_text, y + h - delta_text), (0, 127, 255), 1)
        crop_img = cropping_img.crop((x + delta_text, y + delta_text, x + w - delta_text, y + h - delta_text))
        
        if crop_img:
            if i % 30 == 0:
                # crop_img_name = './cropimg/' + time.strftime("%Y%m%d-%H%M%S") + '.png'
                crop_img_name = './cropimg/cropimg.png'
                crop_img.save(crop_img_name)

                myimg = cv2.imread(crop_img_name)
                crop_img_gray = cv2.cvtColor(myimg, cv2.COLOR_BGR2GRAY)
                binary_img = cv2.threshold(crop_img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                # binary_file_name = './binaryimg/' + time.strftime("%Y%m%d-%H%M%S") + '.png'
                binary_file_name = './binaryimg/binaryimg.png'
                cv2.imwrite(binary_file_name, binary_img)
                text_result = pytesseract.image_to_string(binary_img, lang='eng', config='--psm 6')
                
                text_result = "".join(c for c in text_result if c.isupper())
                if len(text_result) == 1:
                    
                    start_y = y - 2 * delta_color
                    if start_y <= 0:
                        start_y = 0
                        
                    end_y = y - delta_color
                    if end_y >= height:
                        start_y = height
                        
                    start_x = x - 2 * delta_color
                    if start_x <= 0:
                        start_x = 0
                        
                    end_x = x - delta_color
                    if end_x <= width:
                        end_x = width
                    
                    cnt = (end_y - start_y + 1) * (end_x - start_x + 1)
                    pos_x = x - delta_color
                    sum_b, sum_g, sum_r = 0, 0, 0
                    avg_b, avg_g, avg_r = 0, 0, 0

                    for pos_y in range(start_y, end_y):
                        for pos_x in range(start_x, end_x):
                            sum_b = sum_b + color_img[pos_y, pos_x, 0]
                            sum_g = sum_g + color_img[pos_y, pos_x, 1]
                            sum_r = sum_r + color_img[pos_y, pos_x, 2]
                    avg_b = str(int(sum_b/cnt))
                    avg_g = str(int(sum_g/cnt))
                    avg_r = str(int(sum_r/cnt))
                    

                    # calculation of Distance.
                    distance = (default_altitude * image_pixel * cm_per_pixel) / (100*(w + 2*delta_text))
                    text_file_name = './history/' + time.strftime("%Y%m%d-%H%M%S") + '.txt'
                                  
                    f = open(text_file_name, "a")
                    f.write('text : ')
                    f.write(text_result)
                    f.write('\n')
                    f.write('color : ')
                    f.write('[' + avg_b + ',' + avg_g + ',' + avg_r + ']')
                    f.close()
                    
                    distance_list.append(distance)

                    # print(text_result)
                    bounding_file_name = './boundingimg/' + time.strftime("%Y%m%d-%H%M%S") + '.png'
                    cv2.imwrite(bounding_file_name, img)
                    time.sleep(2)
            i += 1
            
        else:
            break
        
        for row, value in enumerate(distance_list, start=0):
            worksheet.write(row, 0, value)
            wb.save('distance.xls')

cap.release()
cv2.destroyAllWindows()