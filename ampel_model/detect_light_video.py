import cv2
import numpy as np

#trafficlight_green_cascade = cv2.CascadeClassifier("trafficlight_green_cascade.xml")
trafficlight_red_cascade = cv2.CascadeClassifier("trafficlight_red_cascade.xml")

cap = cv2.VideoCapture(0) #Camera on
if cap.isOpened()==False:
    cap.open(0)

while True:
    ret, img = cap.read()
    if not ret:
        print("Frame could not be captured")
        break
    #img = cv2.imread('test.jpg', 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #take image into grayscale
   # green = trafficlight_green_cascade.detectMultiScale(gray, 1.3, 5)
    red = trafficlight_red_cascade.detectMultiScale(gray, 1.3, 5)
    #for(x,y,w,h) in green: #draw a rectangle around the detected stop
    #    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2) #color green, line width = 2
    #    font = cv2.FONT_HERSHEY_SIMPLEX #define font for text output
    #    cv2.putText(img, 'green light', (x-5, y-5), font, 1, (0, 255, 0), 3, cv2.LINE_AA) #ampel einblenden
       # print("detectet")
 
    for(x,y,w,h) in red: #draw a rectangle around the detected stop
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2) #color green, line width = 2
        font = cv2.FONT_HERSHEY_SIMPLEX #define font for text output
        cv2.putText(img, 'red light', (x-5, y-5), font, 1, (0, 255, 0), 3, cv2.LINE_AA) #ampel einblenden

        
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): #destroy whindows when pressing q
        break
cap.release()
cv2.destroyAllWindows()
