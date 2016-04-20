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

# FIND AND SHOW CONTOURS OF IMAGE USING ROTATED RECTANGLES
def contourExample(image):
    
    imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cvShow(imgray)
    
    ret,thresh = cv2.threshold(imgray,127,255,0)
    cvShow(thresh)
        
    img,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cvShow(img)
    
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0,0,255),2)
        cvShow(image)
        
      

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
            if cv2.contourArea(cnt) < 10000 and cv2.contourArea(cnt) > 200:
                [x,y,w,h] = cv2.boundingRect(cnt)
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)
                roi = thresh[y:y+h,x:x+w]
                
                # ATTEMPT TO SAVE ONLY CONTOUR AS FEATURES
                #blank = cv2.imread('blankcanvas.jpg')
                #cImg = cv2.drawContours(blank, [cnt], 0, (0,0,0), 1)
                #gray = cv2.cvtColor(cImg,cv2.COLOR_BGR2GRAY)
                #ret,thresh = cv2.threshold(gray,127,255,0)
                #roi = thresh[y:y+h,x:x+w]
                # #####################
                
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
    #local variables
    correct = 0
    total = 0
    response_tuples = []
    removal = []
    
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
        
        # Image Preprocessing
        path = TRAIN_DIR + filename
        image = cv2.imread(path)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray,127,255,0)
        img, contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        
        # OCR
        for cnt in contours:
            if cv2.contourArea(cnt) < 10000 and cv2.contourArea(cnt) > 200:
                #bounding rectangle
                [x,y,w,h] = cv2.boundingRect(cnt)
                
                #cut out same area from threshold image for ocr
                roi = thresh[y:y+h,x:x+w]
                
                # ATTEMPT TO SAVE ONLY CONTOUR AS FEATURES
                #blank = cv2.imread('blankcanvas.jpg')
                #cImg = cv2.drawContours(blank, [cnt], 0, (0,0,0), 1)
                #gray = cv2.cvtColor(cImg,cv2.COLOR_BGR2GRAY)
                #ret,thresh = cv2.threshold(gray,127,255,0)
                #roi = thresh[y:y+h,x:x+w]
                # #####################
                
                roismall = cv2.resize(roi,(20,20))
                roismall = roismall.reshape((1,400))
                roismall = np.float32(roismall)
                #KNN
                retval, results, neigh_resp, dists = model.findNearest(roismall, 5)
                #convert response back to character
                letter = chr(int((results[0][0])))
                #save letter and position data together for post processing
                response_tuples.append((letter, x, y, w, h))
        
        # Remove sub-contours when they are surrounded by a larger contour 
        for i in range(0, len(response_tuples)):
            for j in range(0, len(response_tuples)):
                if i != j:
                    # X and Y coordinates of each bounding rectangle 
                    ix1 = response_tuples[i][1]
                    ix2 = ix1 + response_tuples[i][3]
                    iy1 = response_tuples[i][2]
                    iy2 = iy1 + response_tuples[i][4]
                    jx1 = response_tuples[j][1]
                    jx2 = ix1 + response_tuples[j][3]
                    jy1 = response_tuples[j][2]
                    jy2 = iy1 + response_tuples[j][4]
                    
                    #If response_tuple[i] is surrounded by another response, mark it for removal
                    if ix1 > jx1 and ix2 < jx2 and iy1 > jy1 and iy2 < jy2:
                        removal.append(i)
                        
        # removing...
        indexes = list(set(removal)) # removes duplicate indexes
        indexes = sorted(indexes, reverse=True)
        for x in indexes:
            del response_tuples[x]
                    
        # Sort response based on X-position of contours
        response_sorted = sorted(response_tuples, key=lambda xpos: xpos[1])
        response = ""
        for ltr in response_sorted:
            response = response + ltr[0]
            
        # Print results 
        print "Response: " + response 
        total+=1
        if response == answer: 
            correct+=1
            print "!!"
            
        # Clear lists
        del response_tuples[:]
        del removal[:]
            
    print "Correct: " + str(correct)
        
def main():
    
    #im = cv2.imread('train/apdh.jpg')
    
    #contourExample(im)
    #cvShow(im)
    #canny()
    #train()
    ocr()

if __name__ == '__main__':
    main()
