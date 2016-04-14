import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt


# SHOW IMAGE WITH CV2
def cvShow(image):
    cv2.imshow('image',image)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    #print key

# SHOW IMAGE WITH MATPLOTLIB
def pltShow(image):
    plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])
    plt.show()
    
# CANNY EDGE DETECTION
def canny(image):
    imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    edges = cv2.Canny(thresh, 100, 200)
    plt.subplot(121),plt.imshow(imgray,cmap = 'gray')
    plt.title('Original'),plt.xticks([]),plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Canny Edges'),plt.xticks([]),plt.yticks([])
    plt.show()

# FIND AND SHOW CONTOURS OF IMAGE
def contourExample(image):
    imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cvShow(imgray)
    
    ret,thresh = cv2.threshold(imgray,127,255,0)
    cvShow(thresh)
    
    edges = cv2.Canny(thresh, 100, 200)
    cvShow(edges)
    
    img,contours,hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cvShow(img)
    
    cImg = cv2.drawContours(im3, contours, -1, (0,255,0), 2)
    cvShow(cImg)
    
# FIRST ATTEMPT AT OCR
def ocr(image):
    #for filename in os.listdir(os.getcwd()):
        

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,127,255,0)
    img, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        [x,y,w,h] = cv2.boundingRect(cnt)
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
        roi = thresh[y:y+h,x:x+w]
        cvShow(image)
        #cvShow(roi)
    
def main():
    #cvShow(im)
    #canny(im3)
    #contourExample(im3)
    im = cv2.imread('train/zwyd.jpg')
    ocr(im)

if __name__ == '__main__':
    main()
