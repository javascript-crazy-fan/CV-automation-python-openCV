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


default_altitude, image_pixel = 50, 382
feet2meter = 30.48/100


# file_name = 'ground.png' # from image file

# cap = cv2.VideoCapture(0) # from camera

cap= cv2.VideoCapture('video.mov') # from video file

try:
    os.makedirs('binaryimg')
    os.makedirs('boundingimg')
    os.makedirs('cropimg')
    os.makedirs('frameimg')
    os.makedirs('history')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


def find_marker(image):

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)
 
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key = cv2.contourArea)
	return cv2.minAreaRect(c)


# for writing to excel file
wb = xlwt.Workbook()
worksheet = wb.add_sheet("Sheet 1", cell_overwrite_ok=True)

altitude_list = []

i = 1

while True:
    
    ret, img = cap.read()
    
    if ret == False:
        break
    
    camera_file_name = './frameimg/frameimage.png'
    
    cv2.imwrite(camera_file_name, img)
    
    img = cv2.imread(camera_file_name)
    
    height, width = img.shape[:2]
    
    cropping_img = Image.open(camera_file_name)
    
    
    # img = cv2.imread(file_name)
    # cropping_img = Image.open(file_name)
    
    
    marker = find_marker(img)
    box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
    box = np.int0(box)
    cv2.drawContours(img, [box], -1, (0, 255, 0), 2)
        
    # crop_img = img[(box[2][1] - 100):(box[0][1] + 100), (box[2][0] - 100):(box[0][0] + 100)]
    crop_img = img[(box[2][1]):(box[0][1]), (box[2][0]):(box[0][0])]
    h, w = crop_img.shape[:2]
    
    if i % 5 == 0: 
        # crop_img_name = 'cropimg.png'
        crop_img_name = './cropimg/' + time.strftime("%Y%m%d-%H%M%S") + '.png'
        
        if h > 0.1*height and w > 0.1*width and h < 0.3*height and w < 0.3*width:
            cv2.imwrite(crop_img_name, crop_img)
        else:
            continue
        
        myimg = cv2.imread(crop_img_name)
        crop_img_gray = cv2.cvtColor(myimg, cv2.COLOR_BGR2GRAY)
        binary_img = cv2.threshold(crop_img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        binary_file_name = 'binaryimg.png'
        cv2.imwrite(binary_file_name, binary_img)
        text_result = pytesseract.image_to_string(binary_img, lang='eng', config='--psm 6')
        text_result = "".join(c for c in text_result if c.isupper())
        
        print(text_result)
        time.sleep(2)
    i+=1
        
   
    
    
    
    
    
    
    # img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY)
    # image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
    # ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY) 
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)) 
    # dilated = cv2.dilate(new_img, kernel, iterations=9) 
    # contours, hierarchy = cv2.findContours(image_final,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        
    # # for color detection
    
    # height, width = img.shape[:2] # getting height and width of an image in pixel unit
    # hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # convert BGR color to HSV color space

    # # Low and high color range of Mask for extracting square
    # low = np.array([0, 42, 0])
    # high = np.array([179, 255, 255])
    # mask = cv2.inRange(hsv_frame, low, high)
    # color_img = cv2.bitwise_and(img, img, mask=mask)


    # for contour in contours:
    #     [x, y, w, h] = cv2.boundingRect(contour)
        
    #     if w < 200 or h < 200:
    #         continue

    #     delta_text = 50
    #     delta_color = 20
        
    #     cv2.rectangle(img, (x + delta_text, y + delta_text), (x + w - delta_text, y + h - delta_text), (0, 0, 255), 2)
    #     # crop_img = cropping_img.crop((x + delta_text, y + delta_text, x + w - delta_text, y + h - delta_text))
        
        
    #     if crop_img:
    #         if i % 10 == 0:
    #             # crop_img_name = './cropimg/' + time.strftime("%Y%m%d-%H%M%S") + '.png'
    #             crop_img_name = './cropimg/cropimg.png'
    #             crop_img.save(crop_img_name)
            
    #             myimg = cv2.imread(crop_img_name)
    #             crop_img_gray = cv2.cvtColor(myimg, cv2.COLOR_BGR2GRAY)
    #             binary_img = cv2.threshold(crop_img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #             # binary_file_name = './binaryimg/' + time.strftime("%Y%m%d-%H%M%S") + '.png'
    #             binary_file_name = './binaryimg/binaryimg.png'
    #             cv2.imwrite(binary_file_name, binary_img)
    #             text_result = pytesseract.image_to_string(binary_img, lang='eng', config='--psm 6')
                
    #             text_result = "".join(c for c in text_result if c.isupper())
    #             if len(text_result) > 0:
    #                 text_result = text_result.replace('TI', 'T')
    #                 text_result = text_result.replace('H', 'T')
    #                 text_result = text_result.replace('LB', 'B')
    #                 text_result = text_result.replace('E', 'B')
    #                 text_result = text_result.replace('Q', 'B')
    #                 text_result = text_result.replace('L', 'B')
                    
    #                 start_y = y - 2 * delta_color
    #                 if start_y <= 0:
    #                     start_y = 0
                        
    #                 end_y = y - delta_color
    #                 if end_y >= height:
    #                     start_y = height
                        
    #                 start_x = x - 2 * delta_color
    #                 if start_x <= 0:
    #                     start_x = 0
                        
    #                 end_x = x - delta_color
    #                 if end_x <= width:
    #                     end_x = width
                    
    #                 cnt = (end_y - start_y + 1) * (end_x - start_x + 1)
    #                 pos_x = x - delta_color
    #                 sum_b, sum_g, sum_r = 0, 0, 0
    #                 avg_b, avg_g, avg_r = 0, 0, 0

    #                 for pos_y in range(start_y, end_y):
    #                     for pos_x in range(start_x, end_x):
    #                         sum_b = sum_b + color_img[pos_y, pos_x, 0]
    #                         sum_g = sum_g + color_img[pos_y, pos_x, 1]
    #                         sum_r = sum_r + color_img[pos_y, pos_x, 2]
    #                 avg_b = str(int(sum_b/cnt))
    #                 avg_g = str(int(sum_g/cnt))
    #                 avg_r = str(int(sum_r/cnt))

    #                 distance = (default_altitude * w *  feet2meter) / image_pixel
    #                 text_file_name = './history/' + time.strftime("%Y%m%d-%H%M%S") + '.txt'
                                  
    #                 f = open(text_file_name, "a")
    #                 f.write('text : ')
    #                 f.write(text_result)
    #                 f.write('\n')
    #                 f.write('color : ')
    #                 f.write('[' + avg_b + ',' + avg_g + ',' + avg_r + ']')
    #                 f.close()
                    
    #                 altitude_list.append(distance)

    #                 # print(text_result)
    #                 bounding_file_name = './boundingimg/' + time.strftime("%Y%m%d-%H%M%S") + '.png'
    #                 cv2.imwrite(bounding_file_name, img)
    #         i+=1
            
    #     else:
    #         break
        
    #     for row, value in enumerate(altitude_list, start=0):
    #         worksheet.write(row, 0, value)
    #         wb.save('altitude.xls')

cap.release()
cv2.destroyAllWindows()