import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from sklearn.externals import joblib
from features import Extractor
from scipy.ndimage.measurements import label
from collections import deque
import pickle

def visualize(fig, rows, cols, images, titles):
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.title(i+1)
        plt.axis('off')
        img_dims = len(img.shape)
        if img_dims < 3:
            plt.imshow(img, cmap='hot')
        else:
            plt.imshow(img)
        plt.title(titles[i])

class Tracker:
    def __init__(self):
        model = joblib.load('model.pkl')
        self.svc = model['svc']
        self.X_scaler = model['scaler']
        #self.orient = model['orient']
        self.pixels = model['pixels']
        self.pix_per_cell = model['pixels']
        #self.cells = model['cells']
        #self.spatial_size = model['size']
        #self.hist_bins = model['bins']
        self.history = deque(maxlen=20)

        #self.ystart = 400
        #self.ystop = 680

    def find_and_draw_detections(self, frame):
        vehicles = self.find_cars(frame)
        boxes = np.copy(frame)
        for box in vehicles:
            cv2.rectangle(boxes, (box[0], box[1]), (box[2], box[3]), (0,0,255), 3)
        return boxes

    def process(self, frame):
        vehicles = self.find_cars(frame)
        self.history.append(vehicles)
        labels, heatmap = self.get_labels(frame, frame.shape, 20)
        self.draw_labeled_bboxes(frame, labels)
        self.draw_text(frame, str(labels[1]) + " cars", 20, 20)
        return frame

    def draw_text(self, frame, text, x, y):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)

    def get_labels(self, frame, shape, threshold):
        heatmap = np.zeros((shape[0], shape[1]))

        self.add_heat(heatmap)
        heatmap[heatmap < threshold] = 0
        heatmap = np.clip(heatmap, 0, 255)

        labels = label(heatmap)

        global frame_count
        frame_count += 1
        if False:#frame_count ==  25:
            self.heatmap_visualization(frame, shape, threshold)

        return labels, heatmap

    def heatmap_visualization(self, frame, shape, threshold):
        images = []
        titles = []
        i = 0
        fig = plt.figure(figsize = (12, 12))
        for boxes in self.history:
            heatmap = np.zeros((shape[0], shape[1]))
            for box in boxes:
                heatmap[box[1]:box[3], box[0]:box[2]] += 1
            heatmap = np.clip(heatmap, 0, 255)
            #heatmap[heatmap < threshold] = 0
            vis_heat = np.copy(heatmap / np.max(heatmap) * 255).astype(np.uint8)
            print('maxes', np.max(heatmap), np.max(vis_heat))
            vis_heat = cv2.applyColorMap(vis_heat, cv2.COLORMAP_HOT)
            images.append(vis_heat)
            titles.append("frame " + str(frame_count - i))
            i += 1

        # mapping
        heatmap = np.zeros((shape[0], shape[1]))
        self.add_heat(heatmap)
        heatmap[heatmap < threshold] = 0
        heatmap = np.clip(heatmap, 0, 255)
        images.append(heatmap)
        titles.append("integrated")

        visualize(fig, 7, 3, images, titles)
        plt.show()

        labels = label(heatmap)
        plt.imshow(labels[0], cmap='gray')
        plt.show()

        self.draw_labeled_bboxes(frame, labels)
        plt.imshow(frame)
        plt.show()

    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 4)
        # Return the image
        return img

    def add_heat(self, heatmap):
        for boxes in self.history:
            for box in boxes:
                heatmap[box[1]:box[3], box[0]:box[2]] += 1
        return heatmap

    def find_cars(self, frame):
        scales = [1.25, 1.5, 2.0, 2.5]
        ystarts = [400, 410, 425, 435]
        cars = []
        total = 0
        for ystart, scale in zip(ystarts, scales):
            ystop = ystart + scale * 80
            img = frame[ystart:ystop,:,:]
            #print('strip shape', img.shape)
            #plt.imshow(img)
            #plt.show()

            scale_cars, windows = self.find_cars_scale(img, scale, ystart)
            #print(scale, ': windows ', windows, ' detections ', len(scale_cars))
            #print(scale_cars)
            cars.extend(scale_cars)
            total += windows
        #print('windows: ', total)
        return cars

    def find_cars_scale(self, frame, scale, ystart):
        frame = frame.astype(np.float32) / 255
        img = np.copy(frame)
        (h, w, d) = frame.shape
        frame = cv2.resize(frame, (int(w / scale), int(h / scale)))
        (h, w, d) = frame.shape

        # Define blocks and steps as above
        nxblocks = (w // self.pix_per_cell)-1
        nyblocks = (h // self.pix_per_cell)-1
        #print('shape', frame.shape)
        #print('nblocks', nxblocks, nyblocks)

        window = 64
        nblocks_per_window = (window // self.pix_per_cell)-1
        #print('blocks_per_window', nblocks_per_window)
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        extractor = Extractor(frame)
        hog = [extractor.hog1, extractor.hog2, extractor.hog3]
        #pickle.dump(hog, open('hog.p','wb'))
        #pickle.dump(frame, open('frame.p', 'wb'))

        cars = []
        windows = 0
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                #print(xpos, ypos)
                #print(frame.shape)
                windows += 1

                # features for this patch
                features = extractor.features(xpos, ypos)
                #print('features length', len(features))
                #print('hog shape', extractor.hog1.shape)
                test_features = self.X_scaler.transform(features)
                test_prediction = self.svc.predict(test_features)

                # extract subimage
                if False:
                    xleft = xpos*self.pix_per_cell
                    ytop = ypos*self.pix_per_cell
                    subimage = frame[ytop:ytop + window, xleft:xleft + window]
                    features = Extractor(subimage).features()
                    test_features2 = self.X_scaler.transform(features)
                    test_prediction = self.svc.predict(test_features2)

                    mag = np.linalg.norm(test_features - test_features2)

                    #print('mag', mag)

                if test_prediction == 1:
                    xleft = xpos*self.pix_per_cell
                    ytop = ypos*self.pix_per_cell

                    #print('xb',xb, 'yb', yb )
                    #print('xpos', xpos, 'ypos', ypos)
                    #print('xleft', xleft, 'ytop', ytop)

                    subimage = frame[ytop:ytop + window, xleft:xleft + window]
                    #print(frame.shape)
                    #print(subimage.shape)

                    test_features = self.X_scaler.transform(Extractor(subimage).features())
                    test_prediction = self.svc.predict(test_features)

                    #print("SUBIMAGE Prediction: ", test_prediction)

                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    box = [xbox_left, ytop_draw+ystart, xbox_left+win_draw, ytop_draw + win_draw + ystart]
                    cars.append(box)
        return cars, windows


"""
tracker = Tracker()
frame = mpimg.imread('test_images/test4.jpg')
img = tracker.find_and_draw_detections(frame)
plt.imshow(img)
plt.show()
"""

frame_count = 0
video_name = "test_video"
video_output_name = video_name + '_annotated.mp4'
video = VideoFileClip(video_name + ".mp4")
tracker = Tracker()
video_output = video.fl_image(tracker.process)
video_output.write_videofile(video_output_name, audio=False)

"""
import glob
tracker = Tracker()
images = glob.glob('test_images/*.jpg')
for image in images:
    img = mpimg.imread(image)
    img = tracker.find_and_draw_detections(img)
    plt.imshow(img)
    plt.show()
"""
