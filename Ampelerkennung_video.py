import cv2
from matplotlib import pyplot as plt
import numpy as np
import time

def detect_red_color(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    red = cv2.inRange(hsv, (165,127,127), (175, 255, 255))#Werte evtl anpassen

    target = cv2.bitwise_and(img, img, mask=red)

    return target

def detect_green_color(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    green = cv2.inRange(hsv,(36, 0, 0), (54, 255,255))

    target = cv2.bitwise_and(img, img, mask=green)

    return target

def detect_circles(target_img):
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    _,thresh2 = cv2.threshold(target_gray, 0, 255, cv2.THRESH_BINARY)

    canny2 = cv2.Canny(thresh2,50,100)

    circles = cv2.HoughCircles(canny2, cv2.HOUGH_GRADIENT, dp = 0.2, minDist = 100, param2 = 20, minRadius = 25, maxRadius = 75)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        return circles  

def detect_dark_rectangle(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.medianBlur(img_gray,5)

    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

    rectangle_values = []

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)#0.01?
    
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if len(approx) == 4:
            #Verhältnis abfragen?
            #print("Rectangle detected")
            x, y, w, h = cv2.boundingRect(approx)#minAreaRect()?
            rectangle_values.append([x,y,w,h])

    return rectangle_values #Mittelwerte berechnen?

def detect_traffic_light(rectangle_values, circles):   
    if rectangle_values is not None and  circles is not None:
       
        for x,y,w,h in rectangle_values:
            for x_circle, y_circle, _ in circles:
                if x < x_circle < x + w and y < y_circle < y + h:
                    return True   

    return False

def draw_rectangle(image, rectangle):
    for x, y, w, h in rectangle:
        rectangle_image = cv2.rectangle(image,(x,y),(x+w,y+h),(0, 255, 0),thickness = 5)

    return rectangle_image


def draw_circle(image, circles): 
    for x, y, r in circles:
        circles_image = cv2.circle(image, (x,y),r,(255, 0, 0), thickness = 10)

    return circles_image

def display(images):
    images_rgb = []

    for j in range(len(images)):
        images_rgb.append(cv2.cvtColor(images[j], cv2.COLOR_BGR2RGB)) 

    rows = int(len(images)/2 + 0.5)
    columns = int(len(images_rgb) / rows + 0.5)

    for i in range(len(images)):
        plt.subplot(rows, columns,i + 1),plt.imshow(images_rgb[i],'gray')
        plt.xticks([]),plt.yticks([])
    plt.show()

ampel_rot = False
frame = 1

video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#video.set(cv2.CAP_PROP_FRAME_WIDTH,) # Werte ändern oder Radius und so ändern!
#video.set(cv2.CAP_PROP_FRAME_HEIGHT,240)

time.sleep(1)

start = time.time()

while True:
    ret, image = video.read()
    cv2.imshow('STREAM',image)
    cv2.waitKey(1)
    frame = frame + 1

    if frame > 4:
        break

    #print("On")

    target = detect_green_color(image)
    circles = detect_circles(target)
    ampel_rot = False #also grün

    if circles is None:
        target = detect_red_color(image)
        circles = detect_circles(target)

        if circles is None:
            print("No")
            continue

        else:
            ampel_rot = True  

    rectangles = detect_dark_rectangle(image)

    print(rectangles, circles)
    print()

    detected = detect_traffic_light(rectangles, circles)

    if detected:
        if ampel_rot:
            print("Rote Ampel erkannt!")
        else:
            print("Grüne Ampel erkannt!")    

video.release()

end = time.time()
print()
sek = end - start
print(sek)
print("{0} fps".format(int(frame / (sek))))
cv2.destroyAllWindows()