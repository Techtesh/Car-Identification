# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 15:52:50 2020
 
@author: Hitesh 'Techtesh' 
"""
import cv2
import imutils
import numpy as np
import pytesseract as ptext
ptext.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img=cv2.imread('img1.jpeg')
#img = cv2.resize(img, (620,480) )
#cv2.imwrite('step1.jpg',img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grey scale


cv2.imwrite("ctat2.jpg",gray)

gray = cv2.bilateralFilter(gray, 13, 15, 15)
cv2.imwrite("ctatf.jpg",gray)

edged = cv2.Canny(gray, 30, 200) #Perform Edge detection
cv2.imwrite("step3.jpg",edged)
contours=cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours,key=cv2.contourArea, reverse = True)[:10]
screenCnt = None
for c in contours:
   
    peri = cv2.arcLength(c, True)  #get perimeter of squares
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)#approx pixels to cms
   
    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    detected = 0
    print ("No contour detected")
else:
     detected = 1

if detected == 1:
    cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)
mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(img,img,mask=mask)

(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx+1, topy:bottomy+1]

cv2.imwrite("step4.jpg",Cropped)
img=cv2.imread("step4.jpg")
text=ptext.image_to_string(img)
print(text)

file = open('answer.txt','w') 
file.write(str(text))
