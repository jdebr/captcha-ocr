import sys
import os
import re
import numpy as np
import cv2
from matplotlib import pyplot as plt

TRAIN_DIR = "train/"
PATTERN = re.compile(r'(\D+)\.jpg')

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
    
    cImg = cv2.drawContours(image, contours, -1, (0,255,0), 2)
    cvShow(cImg)
    
# Collect training data by labeling images
def train():

    samples = np.empty((0,400)) 
    responses = []

    for filename in os.listdir(TRAIN_DIR):
        path = TRAIN_DIR + filename
        image = cv2.imread(path)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray,127,255,0)
        img, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) < 10000 and cv2.contourArea(cnt) > 350:
                [x,y,w,h] = cv2.boundingRect(cnt)
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
                roi = thresh[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(20,20))
                cv2.imshow('train',image)
                key = cv2.waitKey(0)
                
                if key == 27:
                    cv2.destroyAllWindows()
                    sys.exit()
                elif key == 32:
                    print "skipped"
                else:
                    responses.append(int(key))
                    sample = roismall.reshape((1,400))
                    samples = np.append(samples,sample,0)
                    
    cv2.destroyAllWindows()
    responses = np.array(responses,np.float32)
    responses = responses.reshape((responses.size,1))
    
    np.savetxt('samples.data',samples)
    np.savetxt('responses.data',responses)
    
def ocr():
    #accuracy reporting
    correct = 0
    total = 0
    
    #training 
    samples = np.loadtxt('samples.data',np.float32)
    responses = np.loadtxt('responses.data',np.float32)
    responses = responses.reshape((responses.size,1))
    
    model = cv2.ml.KNearest_create()
    model.train(samples,cv2.ml.ROW_SAMPLE,responses)
    
    #testing
    for filename in os.listdir(TRAIN_DIR):
        # Determine answer from filename 
        a = PATTERN.match(filename)
        answer = a.group(1)
        print "Answer: " + answer
        
        # Build response 
        response_tuples = []
        
        # Image Preprocessing
        path = TRAIN_DIR + filename
        image = cv2.imread(path)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray,127,255,0)
        img, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        
        # OCR
        for cnt in contours:
            if cv2.contourArea(cnt) < 10000 and cv2.contourArea(cnt) > 350:
                [x,y,w,h] = cv2.boundingRect(cnt)
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
                roi = thresh[y:y+h,x:x+w]
                roismall = cv2.resize(roi,(20,20))
                roismall = roismall.reshape((1,400))
                roismall = np.float32(roismall)
                retval, results, neigh_resp, dists = model.findNearest(roismall, 3)
                letter = chr(int((results[0][0])))
                response_tuples.append((letter, x))
        
        # Sort response based on X-position of contours
        response_sorted = sorted(response_tuples, key=lambda xpos: xpos[1])
        response = ""
        for l in response_sorted:
            response = response + l[0]
            
        print "Response: " + response 
        total+=1
        if response == answer: 
            correct+=1
            
    print "Correct: " + str(correct)
        
def main():
    #cvShow(im)
    #canny(im3)
    
    #im = cv2.imread('train/apdh.jpg')
    #ontourExample(im)
    
    #train()
    ocr()

if __name__ == '__main__':
    main()
