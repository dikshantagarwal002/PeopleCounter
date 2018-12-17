import numpy as np
import cv2 as cv
import imutils

forma="XVID"
fourcc = cv.VideoWriter_fourcc(*forma)

width = 800
height = int(9*width/16)
cap = cv.VideoCapture('solo_person_walking.mp4')
out = cv.VideoWriter('output2.avi', fourcc, 30.0, (width,height))
maybeIn=0

'''
def testIn(x,y):
    rem = 7200-16*y-3*x
    if(rem<0 or rem>-10):
        maybeIn +=1
'''

#fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

while(1):
    grabbed, frame = cap.read()
    if not grabbed:
        break
    frame = imutils.resize(frame, width=width)
    filtered_frame = cv.GaussianBlur(frame, (5, 5), 0)
    fgmask2 = fgbg.apply(filtered_frame)
    
    fgmask = cv.morphologyEx(fgmask2, cv.MORPH_OPEN, kernel)
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel)
    
    thresh = cv.threshold(fgmask, 25, 255, cv.THRESH_BINARY)[1]
    thresh = cv.dilate(thresh, kernel, iterations=15)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    thresh= cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    thresh = cv.erode(thresh, kernel, iterations=14)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    thresh= cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    res = cv.bitwise_and(frame,frame,mask = thresh)
    
    out.write(res)
    cv.imshow("Security Feed", frame)
    cv.imshow("Foreground Mask Readjusted", thresh)
    cv.imshow("Masked Video", res)
    cv.imshow('Foreground Mask',fgmask2)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()