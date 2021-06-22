import cv2
import time
import PoseModule as pm

cap =cv2.VideoCapture("C:/Users/wjdgn/Documents/OpenCV_VisionPractice/PoseEstimationProject/PoseVideos/3.webm")
pTime = 0
detector = pm.poseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.getPose(img)
    print(lmList)
    cTime = time.time()

    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)