import pickle
from features import Extractor
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog

hag = pickle.load(open('hog.p','rb'))
frame = pickle.load(open('frame.p', 'rb'))

pix_per_cell = 8
window = 64
cells_per_step = 2

fram = frame.astype(np.float64)
(h, w, d) = frame.shape

# Define blocks and steps as above
nxblocks = (w // pix_per_cell)-1
nyblocks = (h // pix_per_cell)-1

nblocks_per_window = (window // pix_per_cell)-1
nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
nysteps = (nyblocks - nblocks_per_window) // cells_per_step

xb = 0
yb = 0

ypos = yb*cells_per_step
xpos = xb*cells_per_step
xleft = xpos * pix_per_cell * cells_per_step
ytop = ypos * pix_per_cell * cells_per_step

subimage = frame[ytop:ytop + window, xleft:xleft + window]

plt.imshow(frame)
plt.show()
plt.imshow(subimage)
plt.show()
print(frame.shape)
print(subimage.shape)

extractor1 = Extractor(subimage)
extractor2 = Extractor(frame)

features1 = extractor1.features()
features2 = extractor2.features(xpos, ypos)

bitmap = frame[ytop:ytop + window, xleft:xleft + window] != subimage
zeros = np.nonzero(bitmap)
print('rgbspace', zeros)

frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
subimage = cv2.cvtColor(subimage, cv2.COLOR_RGB2YCrCb)
bitmap = frame[ytop:ytop + window, xleft:xleft + window] != subimage
zeros = np.nonzero(bitmap)
print('ycrcb space', zeros)

orient = 9
pixels = 8
cells = 2

subhog1 = hog(subimage[:,:,0], orientations=orient, pixels_per_cell=(pixels, pixels),
                               cells_per_block=(cells, cells), transform_sqrt=False,
                               feature_vector=False)

framehog1 = hog(frame[:,:,0], orientations=orient, pixels_per_cell=(pixels, pixels),
                               cells_per_block=(cells, cells), transform_sqrt=False,
                               feature_vector=False)

nblocks_per_window = (window // pix_per_cell)-1
subhog_feat1 = subhog1[0:nblocks_per_window, 0:nblocks_per_window]
print('subhog_feat1', subhog_feat1.shape)
framehog_feat1 = framehog1[0:nblocks_per_window, 0:nblocks_per_window]
print('framehog_feat1', framehog_feat1.shape)
bitmap = subhog_feat1 != framehog_feat1
zeros = np.nonzero(bitmap)
print('hog space', zeros)

print(features1[0],features2[0])
mag = np.linalg.norm(features1 - features2)
print('mag', mag)
