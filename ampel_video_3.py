import numpy as np
import time
import cv2


def detect_red_color(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    red = cv2.inRange(hsv, (0,0,255), (255, 40, 255))#Werte evtl anpassen

    target = cv2.bitwise_and(img, img, mask=red)

    return target

def detect_green_color(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    green = cv2.inRange(hsv,(35, 127, 127), (50, 255,255))

    target = cv2.bitwise_and(img, img, mask=green)

    return target

def detect_circles(target_img):
    gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    if ret == False:
      return None

    M = cv2.moments(thresh)

    if M["m00"] != 0:
      cX = int(M["m10"] / M["m00"])
      cY = int(M["m01"] / M["m00"])
    else:
      cX, cY = 0, 0
      return None

    return [cX,cY] 

def draw_circle(image, circles): 
   x,y = circles
   image = cv2.circle(image, (x,y),10,(255, 0, 0), thickness = 2)

def draw_rectangle(image, rectangle, rot):
    if rot:
       farbe = [0,0,255]
    else:
       farbe = [0,255,0]

    x, y, w, h = rectangle
    image = cv2.rectangle(image,(x,y),(x+w,y+h),farbe,thickness = 2)

def detect_dark_rectangle(img, circles):

    x_circle,y_circle = circles

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.medianBlur(img_gray,5)

    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2) 


    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

    rectangle_values = []
    w_values = []

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.1* cv2.arcLength(contour, True), True)
    
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)#minAreaRect()?

            if x < x_circle < x + w and y < y_circle < y + h: 
               rectangle_values.append([x,y,w,h])
               w_values.append(w)
               #print("AMPEL")

    if rectangle_values is None:
       return None
 
    elif len(rectangle_values) == 1:
       index = 0
       #print(index)
       return rectangle_values[index]
       
    elif len(rectangle_values) > 1:
       sorted_w = sorted(w_values)

       for i in range(len(rectangle_values)-1):
          if rectangle_values[i][2] == sorted_w[1]:
             index = i
             #print(index)
             return rectangle_values[index]


#Main
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if cap.isOpened()==False:
    print("ich bin hier")
    cap.open(0)

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Frame could not be captured")
        break
    
    #Bild abschneiden
    height = frame.shape[0]

    width  = frame.shape[1]
    midx = int(width/ 2)

    frame = frame[0:height,midx:width]
    #print(img.shape)
    

    target_green = detect_green_color(frame)
    target_red = detect_red_color(frame)

    #cv2.imshow("green", target_green)
    #cv2.waitKey(1) 

    #cv2.imshow("red", target_red)
    #cv2.waitKey(1) 

    circles_green = detect_circles(target_green)
    circles_red = detect_circles(target_red)

    v = cv2.getTrackbarPos('V','values')
    v = v/ 1000

    if circles_green is None and circles_red is None:
       continue

    final_image = frame.copy()

    if circles_green is not None:
       rectangles_green = detect_dark_rectangle(frame, circles_green)
       if rectangles_green is not None:
          draw_circle(final_image, circles_green)
 
          draw_rectangle(final_image, rectangles_green, False)
          print("Grüne Ampel")
    
    if circles_red is not None:
       rectangles_red = detect_dark_rectangle(frame, circles_red)

       if rectangles_red is not None:
          draw_circle(final_image, circles_red)
          draw_rectangle(final_image, rectangles_red, True)
          print("Rote Ampel")

    cv2.imshow("Ampel", final_image)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()

