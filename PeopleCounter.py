import numpy as np
import cv2 as cv
import imutils

forma="XVID"
fourcc = cv.VideoWriter_fourcc(*forma)
width = 800
height = int(9*width/16)
cap = cv.VideoCapture('moskva.mov')
out = cv.VideoWriter('output.avi', fourcc, 20.0, (width,height))
maybeIn=0

'''
def testIn(x,y):
    rem = 7200-16*y-3*x
    if(rem<0 or rem>-10):
        maybeIn +=1
'''

fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
while(1):
    grabbed, frame = cap.read()
    if not grabbed:
        break
    frame = imutils.resize(frame, width=width)
    fgmask = fgbg.apply(frame)
    
    thresh = cv.threshold(fgmask, 150, 255, cv.THRESH_BINARY)[1]
    thresh = cv.dilate(thresh, None, iterations=2)
    thresh = cv.erode(thresh, None, iterations=2)
    _, cnts, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        # if the contour is too small, ignore it
        if cv.contourArea(c) < 350:
            continue
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv.boundingRect(c)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rectangleCenterPont = ((x + x + w) // 2, (y + y + h) // 2)
        cv.circle(frame, rectangleCenterPont, 1, (0, 0, 255), 5)    
        
        #cv.line(frame, (width, height-180), (0, height-30), (0, 0, 255), 2) #red line
        
        #testIn(rectangleCenterPont)
    cv.line(frame, (width, height-150), (0, height), (250, 0, 1), 2) #blue line
    out.write(frame)
    cv.imshow("Security Feed", frame)
    cv.imshow("Foreground Mask Readjusted", thresh)
    cv.imshow('Foreground Mask',fgmask)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
out.release()
cv.destroyAllWindows()