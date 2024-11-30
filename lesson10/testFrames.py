
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

folder = 'c:/users/stoianov/cv-study/lesson10/data'
files = os.listdir(folder)

for fname in files:
    img = cv2.imread(os.path.join(folder, fname))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print("show",fname)
    plt.imshow(img)
    plt.pause(0.1)
    plt.draw()    
    plt.clf()
