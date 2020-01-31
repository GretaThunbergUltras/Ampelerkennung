import numpy as np
import cv2

def detect_red_color(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    red = cv2.inRange(hsv, (165,127,127), (180, 255, 255))#Werte evtl anpassen

    target = cv2.bitwise_and(img, img, mask=red)

    return target

def detect_green_color(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    green = cv2.inRange(hsv,(35, 127, 127), (50, 255,255))

    target = cv2.bitwise_and(img, img, mask=green)

    return target

def detect_circles(target_img):
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    _,thresh2 = cv2.threshold(target_gray, 0, 255, cv2.THRESH_BINARY)

    canny2 = cv2.Canny(thresh2,50,100)

    cv2.imshow('canny circles', canny2)
    cv2.waitKey(1)
 
    circles = cv2.HoughCircles(canny2, cv2.HOUGH_GRADIENT, dp = 0.2, minDist = 100, param2 = 20, minRadius = 25, maxRadius = 75)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        return circles 


def draw_circle(image, circles):  
    for x,y,r in circles:
        circles_image = cv2.circle(image, (x,y),r,(255, 0, 0), thickness = 10)

    return circles_image


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
    print (frame.shape)
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
        cv2.imshow("Grün", final_image)
        cv2.waitKey(1)

    if circles_red is not None:
        draw_circle(final_image, circles_red)
        cv2.imshow("Rot", final_image)
        cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
