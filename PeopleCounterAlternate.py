import numpy as np
import cv2 as cv
import imutils

width = 800
height = int(9*width/16)
cap = cv.VideoCapture('Escalator.mp4')
line_point1 = (width, height-150)
line_point2 = (0, height)
con=5
people_list = []
inside_count = 0

"""
#maybeIn=0
def testIn(x,y):
    rem = 7200-16*y-3*x
    if(rem<0 or rem>-10):
        maybeIn +=1
"""

ENTERED_STRING = "ENTERED_THE_AREA"
LEFT_AREA_STRING = "LEFT_THE_AREA"
NO_CHANGE_STRING = "NOTHIN_HOMEBOY"

LOWEST_CLOSEST_DISTANCE_THRESHOLD = 20

class Person:

    positions = []

    def __init__(self, position):
        self.positions = [position]

    def update_position(self, new_position):
        self.positions.append(new_position)
        if len(self.positions) > 100:
            self.positions.pop(0)


    def on_opposite_sides(self):
        return ((self.positions[-2][0] > line_point1[0] and self.positions[-1][0] <= line_point1[0])
                or (self.positions[-2][0] <= line_point1[0] and self.positions[-1][0] > line_point1[0]))

    def did_cross_line(self):
        if self.on_opposite_sides():
            if self.positions[-1][0] > line_point1[0]:
                return ENTERED_STRING
            else:
                return LEFT_AREA_STRING
        else:
            return NO_CHANGE_STRING

    def distance_from_last_x_positions(self, new_position, x): #people_list[i].distance_from_last_x_positions(rectangle_center, con)
        total = [0,0]
        z = x
        while z > 0:
            if (len(self.positions) > z):
                total[0] +=  self.positions[-(z+1)][0]
                total[1] +=  self.positions[-(z+1)][1]
            else:
                x -= 1
            z -= 1
        if total[0] < 1 or total[1] < 1:
            return abs(self.positions[0][0] - new_position[0]) + abs(self.positions[0][1] - new_position[1])
        total[0] = total[0] / x
        total[1] = total[1] / x

        return abs(new_position[0] - total[0]) + abs(new_position[1] - total[1])        

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
        
        cv.line(frame, line_point1, line_point2, (250, 0, 1), 2) #blue line
        #cv.line(frame, (width, height-180), (0, height-30), (0, 0, 255), 2) #red line
        
        lowest_closest_distance = float("inf")
        closest_person_index = None
        for i in range(0, len(people_list)):
                if people_list[i].distance_from_last_x_positions(rectangleCenterPont, con) < lowest_closest_distance:
                    lowest_closest_distance = people_list[i].distance_from_last_x_positions(rectangleCenterPont, con)
                    closest_person_index = i
        if closest_person_index is not None:
                if lowest_closest_distance < LOWEST_CLOSEST_DISTANCE_THRESHOLD:
                    people_list[i].update_position(rectangleCenterPont)
                    change = people_list[i].did_cross_line()
                    if change == ENTERED_STRING:
                        inside_count += 1
                    elif change == LEFT_AREA_STRING:
                        inside_count -= 1
                else:
                    new_person = Person(rectangleCenterPont)
                    people_list.append(new_person)
        else:
                new_person = Person(rectangleCenterPont)
                people_list.append(new_person)
        
        #testIn(rectangleCenterPont)
    cv.putText(frame, "Number of people inside: {}".format(inside_count), (10, 20),
	cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv.imshow("Security Feed", frame)
    cv.imshow("Foreground Mask Readjusted", thresh)
    cv.imshow('Foreground Mask',fgmask)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()