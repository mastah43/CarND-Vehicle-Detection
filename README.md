# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


This project implements a software pipeline to detect vehicles in a video   

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## Prerequisites
In order to run the pipeline to produce the project result video at least python 3.6 is needed 
(since e.g. Python type hints are used in the code - see also https://www.python.org/dev/peps/pep-0484/).
It will not run with python 2.
You could also use the file [requirements.txt](requirements.txt) to setup a conda environment with all
the required dependencies but having the following python 3 versions of the packages
numpy, opencv, matplotlib, moviepy installed should be sufficient.

## Files
The following files are contained:
- [project result video](project_video_result.mp4)
- [test result video](test_video_result.mp4)
- [jupyter notebook](vehicle_detection.ipynb)
- output images in folder 'output_images' that are used by this readme

[//]: # (Image References)
[samples_nonvehicles]: ./output_images/samples_nonvehicles.png "Samples Non Vehicles"
[samples_vehicles]: ./output_images/samples_vehicles.png "Samples Vehicles"
[hog_features]: ./output_images/hog_features.png "Hog Features"
[windows_0]: ./output_images/windows-0.jpg "Windows 0"
[windows_1]: ./output_images/windows-1.jpg "Windows 1"
[windows_2]: ./output_images/windows-2.jpg "Windows 2"
[windows_3]: ./output_images/windows-3.jpg "Windows 3"
[windows_4]: ./output_images/windows-4.jpg "Windows 4"
[imgs_test_bboxes_0]: ./output_images/imgs_test_bboxes_0.jpg "Test Image 1 BBoxes"
[imgs_test_bboxes_1]: ./output_images/imgs_test_bboxes_1.jpg "Test Image 2 BBoxes"
[imgs_test_bboxes_2]: ./output_images/imgs_test_bboxes_2.jpg "Test Image 3 BBoxes"
[imgs_test_bboxes_3]: ./output_images/imgs_test_bboxes_3.jpg "Test Image 4 BBoxes"
[imgs_test_bboxes_4]: ./output_images/imgs_test_bboxes_4.jpg "Test Image 5 BBoxes"
[imgs_test_bboxes_5]: ./output_images/imgs_test_bboxes_5.jpg "Test Image 6 BBoxes"
[heatmap]: ./output_images/heatmap.png "Heatmap"


### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

In function 'build_dataset' of the jupyter notebook I am extracting all features of all images in the training data set. 
Specifically in function 'hog_features_for_image' I am extracting HOG features using scikit image function 'hog'.

To speed up the exploration of feature extraction parameters I am reading all training images into memory.

Here are random samples of vehicle images from the training data set.
![alt text][samples_vehicles]

Here are random samples of non vehicle images from the training data set.
![alt text][samples_nonvehicles]

Here is an example using the `YUV` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` 
and `cells_per_block=(2, 2)`:
![alt text][hog_features]

#### 2. Explain how you settled on your final choice of HOG parameters.

I explored various combinations of color spaces, color channel combinations, pixels per cell, cells per block 
and orientations. For each combination I recorded the feature extraction time and the accuracy of the trained 
classifier. 

The following table lists the combinations I explored:

|  No. |  Accuracy | Orientations | Color Space  | Channels  | Pixels per Cell  | Cells per Block | Time Feature Extraction | Time Training |
|---|---|---|---|---|---|---|
|  1 |  0.96 | 9  | HSV  |  2 | 8  | 2  | 22.5 | 5.2 |
|  2 |  0.98 | 9  | HSV  |  0,1,2 | 8  | 2  | 131.7 | 9 |
|  3 |  0.91 | 9  | HLS  |  2 | 8  | 2  | 39.6 | 7.2 |
|  4 |  0.98 | 9  | HLS  |  0,1,2 | 8  | 2  | 72.3 | 19.6 |
|  5 |  0.96 | 9  | YUV  |  0 | 8  | 2  | 36.6 | 5.1 |
|  6 |  0.98 | 9  | YUV  |  0,1,2 | 8  | 2  | 112.4 | 8.3 |
|  7 |  0.96 | 9  | YCrCb  |  0 | 8  | 2  | 35.6 | 5.1 |
|  8 |  0.98 | 9  | YCrCb  |  0,1,2 | 8  | 2  | 114.8 | 8.6 |
|  9 |  0.97 | 11  | YUV  |  0,1,2 | 16  | 2  | 49.3 | 4.5 |

I finally chose a combination with a good trade off between accuracy and feature extraction time as well as the 
number of features in the feature vector to prevent overfitting of the classifier. One problem in the 
training data set is that there are many consecutive images from the same vehicle. By using fewer features
I hope that my classifier generalizes better. E.g. by using only one color channel instead of all three the number
of features is cut to one third.

My final choice on hog parameters is combination no. 9:
TODO
- color space: YUV
- orientation bins: 11
- color channels: all (0,1,2)
- pixels per cell: 16
- cells per block: 2


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using only the HOG features. 
The code is in section 'Train Linear SVM with chosen feature extraction parameters' of the jupyter notebook. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented in section 'Search Vehicles in Windows (HOG once per full image)' of
the jupyter notebook. The function 'find_cars' implements the sliding window search whereas the function
'find_cars_multi_scale_layers' specifies the vertical image boundaries and window scales (of 64 x 64) to be used.

The following images show the windows that where used. Each image is showing a different scale.
The colors green and blue are used so the overlapping rectangles can be distinguished. 
![alt text][windows_0]
![alt text][windows_1]
![alt text][windows_2]
![alt text][windows_3]
![alt text][windows_4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

To optimize the performance of my classifier, I chose to adopt the extraction of hog features once per image 
and window scale since the computation performance of hog feature extraction per sliding window was too low 
to tune the pipeline.

Here are the resulting bounding boxes for the provided test images:
![alt text][imgs_test_bboxes_0]
![alt text][imgs_test_bboxes_1]
![alt text][imgs_test_bboxes_2]
![alt text][imgs_test_bboxes_3]
![alt text][imgs_test_bboxes_4]
![alt text][imgs_test_bboxes_5]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

To filter false positives I used the fact that vehicles in an image should be detected by multiple overlapping
bounding boxes coming from the different scales in the sliding window search. To implement this, I used a heatmap:
Each bounding box for a vehicle detection added 1 to the heat map. I applied a threshold of 1 on this heatmap 
(which means that at least two overlapping bounding boxes are required for a vehicle detection) to derive
the final vehicle detection bounding boxes. 
The code for this part is in section 'Filtering False Positives using Heatmap' in the jupyter notebook.

Here is the heatmap for test image 1:
![alt text][heatmap]


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline would have problems with multiple cars that are very close because they would be detected as a single car.
Due to the training data set used which only contains images from the back of vehicles, crossing vehicles which are 
seen from the side and incoming vehicles might not be detected properly. This could be improved by adding more training 
images from other angles.
Also the bounding boxes detected in a video for the same car differs (wobbles) over time (framewise). This could be
improved by detecting the connected vehicle pixels inside the bounding boxes coming from the labeled heatmap. 
Then a much more precise bounding box could be determined around the vehicle pixels.
