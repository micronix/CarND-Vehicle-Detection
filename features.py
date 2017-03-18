from skimage.feature import hog
import glob
import matplotlib.image as mpimg
import numpy as np
import cv2

class Extractor:
    def __init__(self, img, orient=9, pixels=8, cells=2):
        self.image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        (self.h, self.w, self.channels) = self.image.shape
        self.pixels = pixels
        self.cells = cells
        self.hog_features = []
        self.hog1 = hog(self.image[:,:,0], orientations=orient, pixels_per_cell=(pixels, pixels),
                               cells_per_block=(cells, cells), transform_sqrt=False,
                               feature_vector=False)
        self.hog2 = hog(self.image[:,:,1], orientations=orient, pixels_per_cell=(pixels, pixels),
                               cells_per_block=(cells, cells), transform_sqrt=False,
                               feature_vector=False)
        self.hog3 = hog(self.image[:,:,2], orientations=orient, pixels_per_cell=(pixels, pixels),
                               cells_per_block=(cells, cells), transform_sqrt=False,
                               feature_vector=False)

    # returns the hog features from a sub square image that starts at (x,y)
    def hog(self, xpos, ypos, nblocks_per_window):
        hog_feat1 = self.hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
        hog_feat2 = self.hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
        hog_feat3 = self.hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
        #print('nblocks_per_window', nblocks_per_window)
        #print('xpos', xpos, 'ypos', ypos)
        #print('hog1 shape', hog_feat1.shape)
        hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
        return hog_features

    def bin_spatial(self, img, size=(16, 16)):
        color1 = cv2.resize(img[:,:,0], size).ravel()
        color2 = cv2.resize(img[:,:,1], size).ravel()
        color3 = cv2.resize(img[:,:,2], size).ravel()
        return np.hstack((color1, color2, color3))

    def color_hist(self, img, nbins=32):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def features(self, xpos=0, ypos=0, window=64, pix_per_cell=8):
        xleft = xpos*pix_per_cell
        ytop = ypos*pix_per_cell
        subimage = self.image[ytop:ytop + window, xleft:xleft + window]

        nblocks_per_window = (window // pix_per_cell)-1

        # subsample subimage
        spatial_features = self.bin_spatial(subimage)
        #print('spatial_features shape', spatial_features.shape)

        # histogram of subimage
        hist_features = self.color_hist(subimage)
        #print('hist_features shape', hist_features.shape)

        # hogfeatures of subimage
        hog_features = self.hog(xpos, ypos, nblocks_per_window)
        #print('hog_features shape', hog_features.shape)

        return np.hstack((spatial_features, hist_features, hog_features)).ravel()

if __name__ == "__main__":
    cars = []
    noncars = []

    cars_files = glob.glob('vehicles/*/*.png')
    noncars_files = glob.glob('non-vehicles/*/*.png')

    cars.append(mpimg.imread(cars_files[0]))
    #noncars.append(mpimg.imread())

    cars = np.array(cars)
    noncars = np.array(noncars)

    from features import Extractor
    extractor = Extractor(cars[0])
    print(extractor.features().shape)
