import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
#hands only use RGB img

mpDraw = mp.solutions.drawing_utils


pTime = 0
cTime = 0


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            #landmark give x,y coord for hands
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm) #lm(landmark)에는 xyz 값이 들어있고, xyz는 실제 포지션이 아닌 좌표값(비율)이 들어있으며, 실제 포지션을 구하기 위해서는 실제 width,height값을 곱하여 구한다. 
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                print(id, cx,cy)
                # if id == 0: #0번 인덱스에만 큰 원 그림
                #     cv2.circle(img, (cx,cy), 15, (255,0,255),cv2.FILLED) 

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255),3) #(obj,contents,pos,font,size,color,thickness)

    cv2.imshow("Image", img)
    cv2.waitKey(1)