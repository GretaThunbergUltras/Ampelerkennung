import numpy as np
import cv2
import time

def detect_red_color(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    red = cv2.inRange(hsv, (165,127,127), (175, 255, 255))#Werte evtl anpassen

    target = cv2.bitwise_and(img, img, mask=red)

    return target

def detect_green_color(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    green = cv2.inRange(hsv,(35, 127, 127), (50, 255,255))

    target = cv2.bitwise_and(img, img, mask=green)

   # cv2.imshow('test2', target)
   # cv2.waitKey(10)

    return target

def detect_circles(target_img):
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    _,thresh2 = cv2.threshold(target_gray, 0, 255, cv2.THRESH_BINARY)

    canny2 = cv2.Canny(thresh2,50,100)

    cv2.imshow('test', canny2)
    cv2.waitKey(100)
    indices= np.where(canny2 !=[0])
    coordinates = zip(indices[0], indices[1])
   # circles = cv2.HoughCircles(canny2, cv2.HOUGH_GRADIENT, dp = 1.2, minDist = 100, param2 = 5, minRadius = 10, maxRadius = 34)

   # if circles is not None:
   #     circles = np.round(circles[0, :]).astype("int")
    print(coordinates)
    return coordinates    

def draw_circle(image, circles):  
   #circles_image = 0
   for x, y  in circles:
       circles_image = cv2.circle(image, (x,y),40,(255, 0, 0), thickness = 10)

   return circles_image    

def detect_dark_rectangle(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.medianBlur(img_gray,5)#Fehlerquelle 3

    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)#Fehlerquelle 2 

    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#Fehlerquelle 2

    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]#Fehlerquelle 3

    rectangle_values = []

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)#Fehlerquelle 2
    
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if len(approx) == 4:
            #Verhältnis abfragen?
            #print("Rectangle detected")
            x, y, w, h = cv2.boundingRect(approx)#minAreaRect()?
            rectangle_values.append([x,y,w,h])

    return rectangle_values #Mittelwerte berechnen?

def draw_rectangle(image, rectangle):
    for x, y, w, h in rectangle:
        rectangle_image = cv2.rectangle(image,(x,y),(x+w,y+h),(0, 255, 0),thickness = 5)

    return rectangle_image    

def detect_traffic_light(rectangle_values, circles):   
    if rectangle_values is not None and  circles is not None:
       
        for x,y,w,h in rectangle_values:
            for x_circle, y_circle in circles:
                if x < x_circle < x + w and y < y_circle < y + h:
                    return True   

    return False


start = time.time()

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
    

    target_green = detect_green_color(frame)
    circles_green = detect_circles(target_green)

    target_red = detect_red_color(frame)
    circles_red = detect_circles(target_red)

    final_image = frame.copy()

    if circles_green is None and circles_red is None:
        print("No light found")
        continue

    if circles_green is not None:
        draw_circle(final_image, circles_green)

    if circles_red is not None:
        draw_circle(final_image, circles_red)   

    cv2.imshow('11',final_image)
    if cv2.waitKey(100) & 0xFF == ord('q'): #Wert ändern
        break    

    #Rechtecke
    rectangles = detect_dark_rectangle(frame)

    if not rectangles:
        print("No rectangle found")
        continue

    #cv2.imshow('12',final_image)    
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break 

    else:
        print("Rectangles and circles found")
        draw_rectangle(final_image, rectangles)

    detected = detect_traffic_light(rectangles, circles_green)

    if detected:
        print("Klappt. Grün")

    else:
        detected = detect_traffic_light(rectangles, circles_red)

        if detected:
            print("Klappt. Rot")    

        else:
            print("Keine Ampel gefunden")
            continue    


    # Display the resulting frame
    cv2.imshow('1',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('2',final_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


end = time.time()
sek = end - start
print(sek)