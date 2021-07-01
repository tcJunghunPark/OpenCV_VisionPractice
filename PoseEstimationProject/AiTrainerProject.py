import cv2
import numpy as np
import time
import PoseModule as pm

path = "C:/Users/wjdgn/Documents/OpenCV_VisionPractice/PoseEstimationProject/AiTrainer"
cap = cv2.VideoCapture("C:/Users/wjdgn/Documents/OpenCV_VisionPractice/PoseEstimationProject/AiTrainer/curls.mp4")
# img = cv2.imread("C:/Users/wjdgn/Documents/OpenCV_VisionPractice/PoseEstimationProject/AiTrainer/test.jpg")
pTime = 0
detector = pm.poseDetector()
count = 0
dir = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280,720))

    img = detector.findPose(img,False)
    lmList = detector.getPose(img,False)
    #print(lmList)


    if lmList != None:
        #Left arm
        angle = detector.findAngle(img,11,13,15)
        #right arm
        #detector.findAngle(img,12,14,16)

        per = np.interp(angle, (210,310),(0,100))
        #print(angle, per,"%")
        bar = np.interp(angle, (210,310), (650,100))

        #check the dumbbel curls
        color = (255,0,255)
        
        if per == 100:
            color = (0,255,0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            color = (0,255,0)
            if dir == 1:
                count += 0.5
                dir = 0

        #Draw bar
        cv2.rectangle(img,(1100,100), (1175,650),color, 2)
        cv2.rectangle(img,(1100,int(bar)), (1175,650),color, cv2.FILLED)
        cv2.putText(img,f'{int(per)} %', (1100,75), cv2.FONT_HERSHEY_PLAIN,4,color,4) 
        #Draw Curl

        cv2.rectangle(img,(0,450), (250,720),(0,255,0), cv2.FILLED)
        cv2.putText(img,str(int(count)), (45,670), cv2.FONT_HERSHEY_PLAIN,15,(255,0,0),25) 

        #print(count)
        #cv2.putText(img,str(int(count)), (50,100), cv2.FONT_HERSHEY_PLAIN, 15, (255,0,0),5)
        
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),5) 
    
    cv2.imshow("Image", img)
    cv2.waitKey(10)
