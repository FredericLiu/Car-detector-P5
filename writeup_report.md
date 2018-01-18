
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README

You're reading it!

Note: I implemented this car detectionalgorithm based on Lane-detection program, so the basic structure is as same as Lane-detection program, the main functions are involved in a class called pipeline(), in the pipelineClass.py code file, there are many other functions related to lane-detection, in this program I just put them here, no need to use them.

Besides, the program contains another two major code file:

help_funs.py contains the basic functions like extract features and draw, and a basic sliding window help function.

trainSVC.py implement the feature extraction combination, train SVM, and using a basic sliding window to validate the test result for further programming.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in two files:

1. lines 6-23 of the file called `help_funs.py`), this is the basic funtion to call skimage.feature.hog().

2. in the single_img_features() the in trainSVC.py, hog feature will be extracted as a part of the total feature using for training.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![caliTest](https://github.com/FredericLiu/Car-detector-P5/blob/master/examples/car_not_car.png)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![hog_vis](https://github.com/FredericLiu/Car-detector-P5/blob/master/examples/hog_visualisation.png)

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters, using a basic sliding window on a single test image, shown as below:

![single_image](https://github.com/FredericLiu/Car-detector-P5/blob/master/examples/single_test_image.png)

We can find the combinations of below parameters got the best result:

```python
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```

The total feature vector length is 8460

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the car/non-car small dataset from https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking

Firstly I used StandardScaler to normalize the dataset.

```python
X = np.vstack((car_features, notcar_features)).astype(np.float64)  
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)
```

Then I split the dataset into traing set and test set.

```python
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)
```

Then feed the training data to SVC, and evaluate the test accuracy.

after many experiment, I choose the SVC parameters as C=100, kernel='rbf', gamma='auto', finally got the result as the above picture and:

14.53 Seconds to train SVC...

Test Accuracy of SVC =  0.997727272727

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I use to search windows in multi scale:

```python
ystart = [350,400, 450, 550]
ystop = [400,600,650,720]
scale = [1.0, 1.5, 2, 2.6]
bbox_list = []
draw_img = np.copy(image_undist)
for i in range(len(scale)):
	draw_img, box_list = self.find_cars(draw_img, ystart[i], ystop[i], scale[i])     
	bbox_list.extend(box_list)
```

And the window search algorithm is in code line 433-500 in pipelineClass.py, which is find_cars() function.

This functions calculate HOG only once for each frame, rather than for each window.

The miulti scale window could be visulized by below picture:

![single_image](https://github.com/FredericLiu/Car-detector-P5/blob/master/examples/multi_scale.png)

But there is a problem that since the window search cost too much calculation resource, the processing of each frame is too slow. So the algorithm still need to be refined in the future.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 4 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![example1](https://github.com/FredericLiu/Car-detector-P5/blob/master/examples/example_test1.png)
![example2](https://github.com/FredericLiu/Car-detector-P5/blob/master/examples/example_test2.png)
![example3](https://github.com/FredericLiu/Car-detector-P5/blob/master/examples/example_test3.png)

Note:For these single image tests, I didn't include the heatmap threshold over multi frames. but for video processing, heatmap threshold over multi frames is necesserary.
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://github.com/FredericLiu/Car-detector-P5/blob/master/output_images/lanes_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

And my strategy is integrate heatmap through multi frames (after lots of experiment, I choose 10 frames), for each 10 frames, I threshold the heat of the postion by 9, and during next 10 frames interval, the strategy only draw the box from last detected cars. shown as below code section:

```python
out_img = self.draw_previous_detection(image_undist, self.previous_box_list)

if(self.heat_count == 0):
	self.heat = np.zeros_like(image[:,:,0]).astype(np.float)
self.heat_adding(bbox_list)
self.heat_count += 1
if (self.heat_count == 10):
	self.heat_count = 0
	self.heat_threshold(threshold = 9)  
	out_img, self.previous_box_list = self.draw_thresholded_detection(image_undist)      
```

And I add a small size window to indicate the labled detections.

Here's an example result showing the false detections are filtered out:

![filter_out_false](https://github.com/FredericLiu/Car-detector-P5/blob/master/examples/filter_out_false.png)

From the screen shot you could find on the small label map, on the left side of the frame, there's a false detection, but it was filtered out, wasn't be drawn on the frame.

But there're stll a few false positive on the video the strategy can't filtered out. I tried to increase the number of integrated frame and use higher threshold, but this will also filter out some correct detections. So currently my strategy is just a workaround. the final solution should be training SVM with a much larger dataset.

Here's an example indicating the situation that false postive are failed to be filtered:

![false_detection](https://github.com/FredericLiu/Car-detector-P5/blob/master/examples/false_detection.png)

From the picture, there are two false positive at the bottom of the frame, but only one is filtered out.
---

### Discussion

#### There are two problems in current impementation:

1. the algorithm is too slow to search window in multi scale, I tried to reduce the window number, but this will influence the heatmap filter. So the window search need to be refined further.

2. there are still many flase positves that can't be filtered out, the most effective solution in my opinion should be using a larger dataset to train the model.

