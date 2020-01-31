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

    cv2.imshow('canny circles', canny2)
    cv2.waitKey(1)
    
    indices= np.where(canny2 !=[0])

    if indices is None:
        return None

    x = indices[0]
    y = indices[1]

    #mid_x = np.median(x[not np.isnan(x)])
    #mid_y = np.median(y[not np.isnan(y)])

    mid_x = np.median(x)
    mid_y = np.median(y)

    if np.isnan(mid_x) or np.isnan(mid_y):
        print("Nothing found")
        return None

    mid_x = int(mid_x)
    mid_y = int(mid_y)

    coordinates = [mid_x, mid_y]

    print(coordinates)
 

    return coordinates

def draw_circle(image, circles):  
    x,y = circles
    circles_image = cv2.circle(image, (x,y),40,(255, 0, 0), thickness = 10)

    return circles_image    

def detect_dark_rectangle(img, circles):
    x_circle,y_circle = circles

    img = img[y_circle - 120:y_circle + 120, x_circle - -75:x_circle + 75]

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.medianBlur(img_gray,5)#Fehlerquelle 2

    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)#Fehlerquelle 2 

    cv2.imshow('thresh', thresh)
    cv2.waitKey(1)

    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

    rectangle_values = []

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)#Fehlerquelle 3
    
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
        x_circle, y_circle = circles
       
        for x,y,w,h in rectangle_values:
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
    
    #Bild abschneiden
    height = frame.shape[0]

    width  = frame.shape[1]
    midx = int(width/ 2)


    frame = frame[0:height,midx:width]
    #print(img.shape)

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
        rectangles = detect_dark_rectangle(frame, circles_green)

    if circles_red is not None and rectangles is None:
        draw_circle(final_image, circles_red)  
        rectangles = detect_dark_rectangle(frame, circles_red) 

    cv2.imshow('11',final_image)
    if cv2.waitKey(100) & 0xFF == ord('q'): #Wert ändern
        break    

    #Rechtecke
    #rectangles = detect_dark_rectangle(frame, circles_green)

    if rectangles is None:
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
    cv2.imshow('Ampel original',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('Ampel markiert',final_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


end = time.time()
sek = end - start
print(sek)