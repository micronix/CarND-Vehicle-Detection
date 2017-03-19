# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/cars.png
[image2]: ./output_images/noncars.png
[image3]: ./output_images/hog-visualize1.png
[image4]: ./output_images/hog-visualize2.png
[image5]: ./output_images/hog-visualize3.png
[image6]: ./output_images/hog-visualize4.png
[image7]: ./output_images/hog-visualize5.png
[image8]: ./output_images/windows.png
[image9]: ./output_images/detections1.png
[image10]: ./output_images/detections2.png
[image11]: ./output_images/detections3.png
[image12]: ./output_images/detections4.png
[image13]: ./output_images/detections5.png
[image14]: ./output_images/heatmaps.png
[image15]: ./output_images/labels.png
[image16]: ./output_images/boxes.png
[video1]: ./project_video_annotated.mp4

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in cells 1-3 in the ipython notebook.

- cell 1: imports the needed libraries
- cell 2: trains on 2000 samples, leaving out a validation set allowing us to select the best parameters
- cell 3: helps us visualize how the hog works for different color spaces & hog paramters

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Based on just playing around with the paremeters, I didn't find any significant difference between HSV, HLS, YUV or YCrCb. RGB did consistently worse than the other color schemes. I was not able to get LUV working, I would get an exception but I figured that it was not really necessary to test another color scheme.

The number of orientations did improve but not much past 9 orientations, this was mentioned in the lecture. I decided to stick with 9 orientations. The number of pixels_per_cell determines how much resolution our image will have. With 20 pixels_per_cell we have a fairly low resolution. At 16 pixels_per_cell the accuracy of the classifier dropped of, while at 4 it was fairly high, but would take a lot longer to computer. I decided to go with pixels_per_cell of 8. For cells_per_block I played around and increaing the number of cells_per_block would make finding the hog features calculation faster but didn't increase the accuracy by much, Same with dropping it down. I decided to go with 2 cells_per_block.

Here are some examples with different color spaces and extreme values of orientations, pix_per_cell, and cells_per_block to get a feel of how the paramters work.

Here is an example using the `HSV` color space and HOG parameters of `orientations=10`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

**Vehicle**

![alt text][image3]

**Non-Vehicle**

![alt text][image4]


**Other Examples**

**16 pixels_per_cell**

![alt text][image5]

**4 pixels_per_cell**

![alt text][image6]

**8 Cells per block***

![alt text][image7]

#### 2. Explain how you settled on your final choice of HOG parameters.

Like explained above I settled on the choice of `orientations=9` `pix_per_cell=8` and `cells_per_block=2` based accucary on testing with a sample of 2000 samples and also speed considerations. Although some configurations seemed to score slightly better than others, the amount of time required for the given improvement didn't seemed worth it. These extra features could also cause us to over fit the training data.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in the notebook notebook.ipynb.

- cell 5: extract features from the images and prepare X_train, y
- cell 6: train the SVM classifier
- cell 7: save classifier and scaler to disk

I created a class called Extractor in file features.py used to obtain features from any image. I went with the following features vector:

- spatial features sample each channel by resizing to (16x16)
- color histogram with 16 bins
- hog features with 9 orientations, 8 pix_per_cell, 2 cells_per_block

The Extractor can be used with an image of any size and we can find the features in a subimage of the image by simply passing the pixels and width of the box to sub-sample for example:

```python
# subimage with top left corner at x, y. default size is 64x64 square
features = Extractor(frame).features(x, y)

# training image is already size 64x64 so this will give us the features from
# the whole image
features = Extractor(img).features()
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

At first I implemented the window search shown in class with 4 different scales `scales = [1.25, 1.5, 2.0, 3.3]`. I picked this by looking through the sample images and seeing what were common sizes for vehicles in those images. Searching the entire image was taking too long so I implemented a different methodology in searching noting a few key concepts:

- There is no reason to search for cars at the top of the image in the sky or the trees
- There is no reason to have small windows search lower in the image, because cars will only appear small in the image when they are far away. Therefore we only have to search using small windows in a narrow band near the top of sky in the flat image.
- The lower we are in the image, a larger window makes more sense to search.

We first define the scales to search by and the regions in the image we will search at those scales. The code for this is found in tracker.py lines 132-150. We search at the following scales in the following regions:

| scale | ystart | ystop |
|:--:|:--:|:--|
| 1.25 | 400 | 500 |
| 1.5  | 410 | 530 |
| 2.0  | 425 | 585 |
| 3.3  | 435 | 680 |


The code for the window search at on scale is found in tracker.py lines 152-229. We first resize the image by scaling it down, so that our window size is always 64x64 pixels. Once we resize our image we create an extractor object which stores the whole image hog features. We use a stepping size of 2 cells_per_step for our window search.

The sliding window search searches only a narrow band between pixel posision `400` and `680` vertically.

Below are all the search windows at different sizes:

![alt text][image8]

This means we be searching 165 windows.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As stated above I used 4 different scales, each scale searches horizontally with different offsets.

I tried playing with the C parameter in the svm, which is how much penalty to give erros. I used the default L2 norm as well. Here are some example images with windows drawn when the classifier identifies a vehicle.

![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]

Our classifer does make identify false positives, but we should be able to remove these with the heatmap method.

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_annotated.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

What gave me the best results was to save the last 20 heatmap frames. Then in the current frame, for each pixel I added the value of the last 20 heatmaps and had a threshold of 20.

Below you can see the previous 20 frames and then the last image shows the integration of all those frames:

![alt text][image14]

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all twenty frames:

![alt text][image15]

Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][image16]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Initially my classifier was performing badly once I did a window search on the whole image. The problem was that I was not using the YCrCb color space on the whole image when using hog features. It really helps to have a nice color space.

Once I got fixed my issue, the amount of time spent searching was more than a second per image. On my laptop this equated to about 25 minutes to process the video and I was only doing 2 scales. Finding a more efficient way of doing the window searched helped to try different parameters. What really helped her was not to search using small windows lower in the frame.

This pipeline will probably fail when there are more extreme lighthing conditions. Also, just looking at a few example vehicles, most cars are either sedans or suvs. I did not see any examples of big trucks, fire trucks, and in general weird cars. This is more a limitation of the training data, but I would expect it fail in this cases.

A big failure case, that would have to be addressed as soon as possible is that in our example video, there was a barrier blocking cars going in the opposite direction, I image that this simplified the problem because our training data only needed to show examples of the backs of cars. If we were driving on a lane without a divider I imagine that we would need more training examples, but we would also probably run into more problems with false negatives.

One problem that I was not able to address in this project was some small false positives that appeared in the middle of the frame, and also when one car passed in front of another car, our system assumed that the other car seased to exist. It would probably be beneficial to implement something that would try to keep track of how many cars we are currently seeing. We know that a car can't just appear out of nowhere, so we detect when a car enters the field of vision from the edges of the image. If we "detect" a car for the first time, in the middle of the image, it is probably a false positive. Once the car moves to the edge of the car moves to the edge of the screen and disappears we can assume that the car is no longer visible.
