## Writeup Report

---

**Project 5: Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/Car_NonCar.png
[image2]: ./output_images/Car_NonCar_Hog_Feature.png
[image3]: ./output_images/grid1high.png
[image4]: ./output_images/grid1low.png
[image5]: ./output_images/grid11.png
[image6]: ./output_images/grid12.png
[image7]: ./output_images/grid15.png
[image8]: ./output_images/grid2.png
[image9]: ./output_images/grid3.png
[image10]: ./output_images/big_grid.png
[image11]: ./output_images/process_all_test_img.png
[image12]: ./output_images/test5_window_search_result.png
[image13]: ./output_images/test5_window_search_heat.png
[image14]: ./output_images/thresh_heat_test5.png

[video1]: ./project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Extract HOG features from the training images.

The code for this step is contained in the code cell **In5** of the IPython notebook (shown below). This function originally came from class materials.

```python
# Define a function to return HOG features and visualization
# This function will be called in extract_features function
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=feature_vec,
                                  block_norm="L2-Hys")
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec,
                       block_norm="L2-Hys")
        return features
```

Before this, I first started by reading in all the `vehicle` and `non-vehicle` images.  Here is a few examples of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space G channel and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. My final choice of HOG parameters.

I tried various combinations of parameters and tested them with LinearSVC training, got the following results (training using HOG only):

colorspace = 'RGB'  orient = 9   pix_per_cell = 8   cell_per_block = 2   hog_channel = 'ALL': **55.7 Seconds to train, accuracy 0.9457**
colorspace = 'HSV'  orient = 9   pix_per_cell = 8   cell_per_block = 2   hog_channel = 'ALL': **34.6 Seconds to train, accuracy 0.9682**
colorspace = 'YUV'  orient = 9   pix_per_cell = 8   cell_per_block = 2   hog_channel = 'ALL': **55.7 Seconds to train, accuracy 0.9457**
colorspace='YCrCb'  orient = 9   pix_per_cell = 8   cell_per_block = 2   hog_channel = 'ALL': **21.8 Seconds to train, accuracy 0.9626**

I was not satisfied with the accuracy, so I enabled spatial features and color histogram features, together with YCrCb HOG features. The parameters I finally got are:

```python
colorspace = 'YCrCb'
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL'
spatial_feat=True
hist_feat=True
spatial_size=(32, 32)
hist_bins=32
```

And I got **10.79 seconds of training time and accuracy of 0.9899.** I chose this set of parameters because the accuracy is high enough and the short training time is good for quick tuning.

#### 3. Train a classifier using selected features.

As stated in the previous point, I trained a linear SVM and got 0.9899 accuracy. The code is as follows:

```python
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts:      ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
```

Actually I also tried the **GridSearchCV** function to try more SVM parameters. But the training time was 25 minutes and I got an accuracy of 0.9742. So this was not so attractive compared with a simle linear SVM.

### Sliding Window Search

#### 1. Implement a sliding window search.  Tuned scales of window and start/stop y coordinate to optimize.

I decided to use small windows to search far road surfaces and big window to search near. And I also tuned the searching start y coordinate for windows of each size:

Small windows (scale=1):

![alt text][image3]

![alt text][image4]

A little bigger windows (scale=1.1, 1.2 and 1.5):

![alt text][image5]

![alt text][image6]

![alt text][image7]

Even bigger windows (scale = 2, 3 and 3.5):

![alt text][image8]

![alt text][image9]

![alt text][image10]

The basic idea is to cover all the interested area with correct sized grids and minimize searching time.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Actually I struggle a lot (a few days of tuning and re-training) until I finally got some satisfying results. I maining tuned the Feature Extraction parameters and searching window size/position.

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, using the window sizes and positions explained in the previous section, which provided a nice result.  Here are some example images:

![alt text][image11]
---

### Video Implementation

#### 1. The link to my final video output.  The entire pipeline performs reasonably well on the entire project video.

Here's a [link to my video result](https://youtu.be/BbHzG2za7WA). It can also be found as project_video_out.mp4 in my repo.


#### 2. I implemented heat threshold filter for false positives and `scipy.ndimage.measurements.label()` method for combining overlapping bounding boxes. However, this was not enough for processing video. I also implemented a Heat-Map buffer filter that takes into account history of Heat Maps of 3 consecutive video frames.

I recorded the positions of positive detections in each test image.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  
This is done by the following functions from class materials:

```python
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
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
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img
```

Here's an example result showing the detected positive bounding boxes and corresponding heatmap from test image test5.jpg, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the image:

### Here are positive bounding boxes:

![alt text][image12]

### Here is the output of `apply_threshold` function on the integrated heatmap:
![alt text][image13]

### Here the resulting heat map and bounding boxes that are drawn onto the original image:
![alt text][image14]

### To successfully get rid of false positives in the video, I also implemented an algorithm to accumulate 3 consecutive frames and apply threshold on the sum of heat maps of the 3 latest video frames. I also tuned the threshold again for this feature.

The code is at the end of my pipeline:

```python
class HeatHistory():
    def __init__(self):
        self.history = []
```

and:

```python
    # consider history frames for detection
    #history = HeatHistory()
    
    heat = np.zeros_like(img[:,:,0])
    heat = add_heat(heat, rectangles)
    
    if len(history.history) >= 3:
        history.history = history.history[1:]
    
    history.history.append(heat)
    heat_history = reduce(lambda h, acc: h + acc, history.history)/3
    
    heat = apply_threshold(heat_history, 4)
     
    labels = label(heat)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
```

This also improved the pipeline performance on the video.

---

### Discussion

#### Difficulties on dealing with false positives.

Apperantly a lot of work can be done in the future to improve. However, during this project, the problem that I spent most of my time and effort is dealing with false positives. This is very tricky. Here are some methods that I have tried (I actually tried all of them):

#### Taking spatial and color histogram features for training SVM
This part has been illustrated in previous sections. I got nearly 99% accuracy. Maybe I can try some other parameter combination, but I doubt if it can make a significant difference. Perhaps this is worth exploring.

#### Tuning searching window size and position
At first I tried to tune  window size and position, and it worked. But I guess I can only improve that much with tuning these parameters.

#### Implementing a buffer of heat map history to filter false positives
This is proven to be working well.

#### Hard Data Mining
I manually grabbed areas in the video which seemed to cause false positives, and dumped them into the Non-Vehicle data set, and re-trained the classifier.
