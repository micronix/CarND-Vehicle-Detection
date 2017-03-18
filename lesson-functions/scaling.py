import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from sklearn.externals import joblib


ystart = 400
yend = 680

frame = mpimg.imread('test.jpg')

subimage = frame[ystart:yend,:,:]
subimage = subimage.astype(np.float32) / 255
(h, w, d) = subimage.shape

pixels = 8

scales = [3.33, 2, 1.5, 1.25]
offset = [0.1, 0.1, 0.1, 0.1]
color = [(0,0,1.0), (0,1.0,0), (1.0, 0, 0), (1.0, 1.0, 0)]
color = [(0,0,255), (0,255,0), (255, 0, 0), (255, 255, 0)]
for scale, offset, color in zip(scales, offset, color):
    img = cv2.resize(subimage, (int(w / scale), int(h / scale)))
    (h, w, d) = img.shape
    ytop = h * offset

    window = 64
    nxsteps = (w - window) // pixels
    nysteps = (h - window) // pixels

    for xb in range(nxsteps):
        xleft = xb * pixels
        box = np.array([xleft, ytop, xleft + window, ytop + window]).astype(np.float32)
        #cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
        box = box * scale
        box[1] += ystart
        box[3] += ystart
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

    #plt.imshow(img)
    #plt.show()

plt.imshow(frame)
plt.show()
