import numpy as np
import cv2

face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye=cv2.CascadeClassifier('haarcascade_eye.xml')
cap=cv2.VideoCapture(0)
while True:
    ret, img=cap.read()
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(gray)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y), (x+y, y+h), (255,0,0), 2)
        roi_gray=gray[y: y+h, x: x+w]
        roi_color=img[y: y+h, x: x+w]
        eyes=eye.detectMultiScale(roi_gray)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey), (ex+ey, ey+eh), (0,250,0), 2)
    cv2.imshow('Face', img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
