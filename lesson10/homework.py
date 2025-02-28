import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

### Step 1
#Decide what video you are going to use for this homework, select an object and generate the template. You can use any video you want (your own, from Youtube, etc.)
#and track any object you want (e.g. a car, a pedestrian, etc.).
print(cv2.__version__)
stream = cv2.VideoCapture("d:\RepoDevel\cv-study\lesson10\istockphoto-1155019062-640_adpp_is.mp4")

# target appears on 5th frame
for _ in range(5):
	ret, frame = stream.read()
 
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

tgtRect = [665,270,760,340]
tgtBbox = [tgtRect[0], tgtRect[1], tgtRect[2]-tgtRect[0], tgtRect[3]-tgtRect[1]]

### Step 2
#Initialize a tracker (e.g. KCF).

trackerKCF = cv2.TrackerKCF_create()
trackerKCF.init(frame, tgtBbox)
trackerCSRT = cv2.TrackerCSRT_create()
trackerCSRT.init(frame, tgtBbox)

cv2.rectangle(frame, tgtRect[:2], tgtRect[2:], (0, 255, 0), 3)

if 1:
	plt.imshow(frame)
	plt.draw()    
	plt.waitforbuttonpress(1.5)

def drawBox(img, bbox, clr):
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), clr, 2)

### Step 3
#Run the tracker on the video and the selected object. Run the tracker for around 10-15 frames.
kcfTime = 0
csrtTime = 0

while True:
    ret, frame = stream.read()
    
    if not ret:
        print("EOF reached")
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    startTime = time.perf_counter()
    kcfResult, kcfBbox = trackerKCF.update(frame)
    kcfTime = kcfTime + time.perf_counter()-startTime

    startTime = time.perf_counter()
    csrtResult, csrtBbox = trackerCSRT.update(frame)
    csrtTime = csrtTime + time.perf_counter()-startTime
         
    if kcfResult:
        drawBox(frame, kcfBbox, (0, 255, 0))

    if csrtResult:
        drawBox(frame, csrtBbox, (255, 0, 0))
        
    plt.imshow(frame)
    plt.draw()    
    plt.waitforbuttonpress(0.01)
    plt.clf()

print("KCF time:", kcfTime)
print("CSRT time:", csrtTime)

# по результатам может уверенно сказать, что CSRT (красный бокс) уверенно лучше справляется 
# с трекингом обьектов которые меняют свои размеры, при этом более ресурсоемкий, в моем случае в 8 раз (0.53с против 4.1с)