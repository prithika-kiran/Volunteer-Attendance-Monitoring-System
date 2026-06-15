import cv2
from datetime import datetime
import csv
import os

name = input("Enter Volunteer Name: ")

haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%d-%m-%Y")
    time = now.strftime("%H:%M:%S")
    file_exists = os.path.exists("attendance.csv")
    with open("attendance.csv", "a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Name", "Date", "Time"])
        writer.writerow([name, date, time])
        
attendance_marked = False
        
while True:
    _, img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        if not attendance_marked:
            mark_attendance(name)
            attendance_marked = True
            print(f"{name} marked present")
        cv2.rectangle(img,(x, y),(x+w, y+h),(0,255,0),2)
    cv2.imshow("Volunteer Attendance", img)
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
