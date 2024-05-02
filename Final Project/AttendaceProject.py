import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path='ImagesAttendance'
images=[]
classNames=[]
myList=os.listdir(path)
print(myList)
for cl in myList:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    today = datetime.now().strftime('%Y-%m-%d')
    with open('Attendance.csv', 'a+') as f:
        f.seek(0)
        for line in f.readlines():
            if today in line and name in line:
                return  # If the person is already marked present for today, exit the function
        now = datetime.now().strftime('%H:%M:%S')
        f.write(f'{name},{now},{today}\n')



encodeListKnown=findEncodings(images)
print('Encoding Complete')

cap=cv2.VideoCapture(1)
start_time = cv2.getTickCount()


while True:
    success,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesCurFrame=face_recognition.face_locations(imgS)
    encodeCurFrame=face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex=np.argmin(faceDis)


        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1=faceLoc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
    cv2.imshow('Webcam',img)
    k=cv2.waitKey(1)
    if k==ord('q') or (cv2.getTickCount() - start_time) / cv2.getTickFrequency() > 60:
        break
cap.release()
cv2.destroyAllWindows()