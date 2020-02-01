import cv2
import numpy as np
import brickpi3

BP = brickpi3.BrickPi3()

cap = cv2.VideoCapture(0) #activate camera

trafficlight_green_cascade = cv2.CascadeClassifier("trafficlight_green_cascade.xml") #load cascade file for green trafficlight
trafficlight_red_cascade = cv2.CascadeClassifier("trafficlight_red_cascade.xml") #load cascade file for red trafficlight

BP.set_motor_power(BP.PORT_B, 30) #start driving

while True:
    img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #image to grayscale
    tl_r = trafficlight_red_cascade.detectMultiScale(gray, 3, 5) #detect red trafficlight
    tl_g = trafficlight_green_cascade.detectMultiScale(gray, 3, 5) #detect green trafficlight

    for(x,y,w,h) in tl_g: #draw rectangle around the detected trafficlight
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2) #color green, line width = 2
        font = cv2.FONT_HERSHEY_SIMPLEX #define font for text output
        cv2.putText(img, 'green Trafficlight', (x-5, y-5), font, 1, (0, 255, 0), 3, cv2.LINE_AA) #write trafficlight when detected
        print("Detected green Trafficlight")

    for(x,y,w,h) in tl_r: #draw rectangle around the detected trafficlight
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2) #color red, line width = 2
        font = cv2.FONT_HERSHEY_SIMPLEX #define font for text output
        cv2.putText(img, 'red Trafficlight', (x-5, y-5), font, 1, (255, 0, 0), 3, cv2.LINE_AA) #write trafficlight when detected
        print("Detected red Trafficlight")
    
    if len(tl_r) !=0:
       BP.set_motor_power(BP.PORT_B, 0)
       print ("Stopping...")
    elif len(tl_g) != 0:
       BP.set_motor_power(BP.PORT_B, 30)
       print ("Driving...")
        
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): #destroy whindows when pressing q
       BP.set_motor_power(BP.PORT_B, 0)
       break
cap.release()
cv2.destroyAllWindows()