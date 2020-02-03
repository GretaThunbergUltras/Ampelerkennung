import numpy as np
import time
import cv2
from botlib.motor import Motor
from botlib.bot import Bot
import brickpi3

APPROACH_POWER, DEFAULT_POWER = 30, 30
RIGHT_MIN, RIGHT_MAX = 20, 40
FRONT_MIN, FRONT_MAX = 0, 30
COLLECT_TIMES = 5

def __init__(self):
    self._bot = Bot()
    self._bot.calibrate()
    
def follow_line(self):
    from multiprocessing import Process

    # Dieser endlose Iterator liefert Lenkunswerte
    # um auf der Linie zu bleiben.
    linetracker = self._bot.linetracker()
    self._track_paused = False

    def follow():
        for improve in linetracker:
            if improve != None:
                self._bot.drive_steer(improve)
                sleep(0.1)

            # Wenn Line Tracking angehalten wurde gehe
            # in eine Schleife
            # TODO: Das ist ziemlich ineffizient
            while self._track_paused:
                sleep(0.1)

    self._track_thread = Process(group=None, target=follow, daemon=True)
    self._track_thread.start()

def detect_red_color(img): #Rote Farbe (über Helligkeit) erkennen

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    red = cv2.inRange(hsv, (0, 0, 255), (255, 40, 255))

    target = cv2.bitwise_and(img, img, mask=red)

    return target

def detect_green_color(img): #Grüne Farbe erkennen

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    green = cv2.inRange(hsv, (35, 127, 127), (50, 255, 255))

    target = cv2.bitwise_and(img, img, mask=green)

    return target

def detect_circles(target_img): #Mittelpunkte der Konturen berechnen und zurückgeben

    gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

    if ret == False:
        print("falsch")
        return None

    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    x_values = []
    y_values = []
    if cnts is None:
        print("nichts")
        return None
    for c in cnts:
        M = cv2.moments(c)
        
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            continue
        x_values.append(cX)
        y_values.append(cY)

    xy = list(zip(x_values, y_values))
    return xy

def draw_rectangle(image, rectangle, rot): #Rechteck in der jeweiligen Farbe zeichnen
    if rot:
        farbe = [0,0,255]
    else:
        farbe = [0,255,0]

    x, y, w, h = rectangle
    image = cv2.rectangle(image, (x,y), (x+w,y+h), farbe, thickness = 2)

def detect_dark_rectangle(img, circles): #Dunkles Rechteck erkennen, indem ein Kreismittelpunkt liegt 
    #und das eine Breite besitzt, die den Erfahrungswerten für einen Abstand zur Ampel von ungefähr 10cm entspricht 

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.medianBlur(img_gray, 5)

    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) 


    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rectangle_values = []
    w_values = []

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.1* cv2.arcLength(contour, True), True)
    
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)

            for x_circle, y_circle in circles:
                if x < x_circle < x + w and y < y_circle < y + h and w > 10 and w < 40:
                    rectangle_values.append([x,y,w,h])
                    w_values.append(w)

    if rectangle_values is None:
        return None

    w_sorted = sorted(w_values)

    for i in range(len(rectangle_values)):
        if len(rectangle_values) == 1:
            return rectangle_values[0]
        if rectangle_values[i][2] == w_sorted[1]:
            return rectangle_values[i]

#Main

BP = brickpi3.BrickPi3()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if cap.isOpened()==False: #Überprüfen, ob Kamera bereit ist
    print("ich bin hier")
    cap.open(0)

#Motor._bp.set_motor_power(BP.PORT_B, 20)
self.follow_line()

start = time.time()

while(cap.isOpened()):
    end = time.time()

    if end - start > 20: #nach 20 Sekunden abbrechen
        break

    ret, frame = cap.read()
    if not ret:
        print("Frame could not be captured")
        break
    
    height = frame.shape[0]

    width  = frame.shape[1]
    midx = int(width/ 2)

    frame = frame[0:height, midx:width] #Nur auf rechte Seite des Bildes fokussieren
    

    target_green = detect_green_color(frame)
    target_red = detect_red_color(frame)

    circles_green = detect_circles(target_green)
    circles_red = detect_circles(target_red)

    if not circles_green and not circles_red:
        continue

    final_image = frame.copy()

    if circles_green:
        rectangles_green = detect_dark_rectangle(frame, circles_green)

        if rectangles_green is not None: #Grünes Licht in Rechteck erkannt -> geradeaus fahren
            green_image = frame.copy()
 
            draw_rectangle(green_image, rectangles_green, False)
            final_image = green_image
            print("Grüne Ampel")
#            Motor._bp.set_motor_power(BP.PORT_B, 20)
    
    if circles_red:
        rectangles_red = detect_dark_rectangle(frame, circles_red)

        if rectangles_red is not None: #Rotes Licht in Rechteck erkannt -> Anhalten
            red_image = frame.copy()
          
            draw_rectangle(red_image, rectangles_red, True)
            final_image = red_image

            print("Rote Ampel")
#            Motor._bp.set_motor_power(BP.PORT_B, 0)

    cv2.imshow("Ampel", final_image) 
    cv2.waitKey(1)

#Motor._bp.set_motor_power(BP.PORT_B, 0)

cap.release()
cv2.destroyAllWindows()
