### Step 1
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

stream = cv2.VideoCapture("17000628-hd_1920_1080_60fps.mp4")

ret, frame = stream.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

tgtRect = [1050,70,1160,320]
tgtBbox = [tgtRect[0], tgtRect[1], tgtRect[2]-tgtRect[0], tgtRect[3]-tgtRect[1]]

### Step 2
trackerKCF = cv2.TrackerKCF_create()
trackerKCF.init(frame, tgtBbox)
trackerCSRT = cv2.TrackerCSRT_create()
trackerCSRT.init(frame, tgtBbox)

#draw initial frame
cv2.rectangle(frame, tgtRect[:2], tgtRect[2:], (0, 255, 0), 3)
plt.imshow(frame)
plt.show(), plt.draw()    
plt.waitforbuttonpress(0.1)
plt.clf()


#process all video
def drawBox(img, bbox, clr):
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), clr, 2)
    
while False:
#for i in range(2):
    ret, frame = stream.read()
    
    if not ret:
        print("EOF reached")
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    kcfResult, kcfBbox = trackerKCF.update(frame)
    csrtResult, csrtBbox = trackerCSRT.update(frame)
    if kcfResult:
        drawBox(frame, kcfBbox, (0, 255, 0))

    if csrtResult:
        drawBox(frame, csrtBbox, (255, 0, 0))
        
    plt.imshow(frame)
    plt.show(), plt.draw()    
    plt.waitforbuttonpress(0.1)
    plt.clf()
    time.sleep(0.1)
