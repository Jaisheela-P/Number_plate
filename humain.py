import numpy as np
import cv2
import  imutils
import  pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
input_image = cv2.imread('testing.jpeg')
input_image = imutils.resize(input_image, width=500)
cv2.imshow("Original Image", input_image)
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)# grayscale conversion
cv2.imshow("Grayscale Conversion image", gray_image)
edge = cv2.Canny(gray_image, 170, 200)# edges detection image
cv2.imshow("Canny Edges image", edge)
(cnts, _) = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)# finding contours using the edges
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
Plate_count = None
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  
            Plate_count = approx 
            break
cv2.drawContours(input_image, [Plate_count], -1, (0,255,0), 3)
cv2.imshow("Final Image With Number Plate Detected", input_image)
cv2.imwrite('D:\python\images.png',input_image)
text = pytesseract.image_to_string(Image.open('images.png'))
print(text)
