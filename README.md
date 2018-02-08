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

### Training Data Set
The pipeline expects the training data set in folder ./training_dataset with vehicle images 
in ./training_dataset/vehicles and non vehicle images in ./training_dataset/non-vehicles.
The image files are expected to have a size of 64 x 64 pixels.
You could use the udacity provided data set: 
* https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip
* https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip

### Python 3.6
In order to run the pipeline to produce the project result video at least python 3.6 is needed 
(since e.g. Python type hints are used in the code - see also https://www.python.org/dev/peps/pep-0484/).
It will not run with python 2.

## Python Modules
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
[color_histogram]: ./output_images/color_histogram.png "Color Histogram"
[windows_0]: ./output_images/windows-0.jpg "Windows 0"
[windows_1]: ./output_images/windows-1.jpg "Windows 1"
[windows_2]: ./output_images/windows-2.jpg "Windows 2"
[windows_3]: ./output_images/windows-3.jpg "Windows 3"
[windows_4]: ./output_images/windows-4.jpg "Windows 4"
[windows_5]: ./output_images/windows-4.jpg "Windows 4"
[windows_6]: ./output_images/windows-4.jpg "Windows 4"
[windows_7]: ./output_images/windows-4.jpg "Windows 4"
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

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` 
and `cells_per_block=(2, 2)`. I used each channel of this color space with these parameters as features for the 
vehicle detection. Especially the 'Y' channel was clearly differentiating vehicle from non vehicle images.
This can be seen in the following plot:
![alt text][hog_features]

#### 2. Explain how you settled on your final choice of HOG parameters.

I explored various combinations of color spaces, color channel combinations, pixels per cell, cells per block 
and orientations. For each combination I recorded the feature extraction time and the accuracy of the trained 
classifier. The exploration approach and results are in section 'Parameter Exploration for SVM Training' 
of the jupyter notebook.

I finally chose a combination with a good trade off between accuracy and feature extraction time as well as the 
number of features in the feature vector to prevent overfitting of the classifier. One problem in the 
training data set is that there are many consecutive images from the same vehicle. By using fewer features
I hope that my classifier generalizes better. E.g. by using only one color channel instead of all three the number
of features is cut to one third.

My final choice on hog parameters is:
- color space: YCrCb
- orientation bins: 8
- color channels: all (index 0,1 and 2)
- pixels per cell: 8
- cells per block: 2

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the HOG featuresa and color histogram features.

For the color histogram I am using 32 bins for every channel in RGB color space.
I am using all channels of RGB for the color histogram since the histogram of vehicle images differentiates 
well from non vehicle images. I explored parameters for color histograms - other colorspace like Y of YUV also work 
well. I decided to use all RGB channels in order to have a significant number of features from color histograms
compared to number of hog features per image.

The following plot shows the difference in RGB histograms for a vehicle and a non vehicle image.

![alt text][color_histogram]
 
The code for SVM training is in section 'Train Linear SVM with chosen feature extraction parameters' of the jupyter notebook.
I augmented the training data set by flipping all training images to improve generalization of the classifier.
Also I am scaling the combined features of HOG and color histogram using a scikit StandardScaler.
I used 80% of the shuffled data set for training and 20% for testing.
My classifier achieved an accuracy of 0.9894 and uses feature vectors of length 4800 (hog features + color histograms)
as an input.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented in section 'Search Vehicles in Windows (HOG once per full image)' of
the jupyter notebook. The function 'find_cars' implements the sliding window search whereas the function
'find_cars_multi_scale_layers' specifies the vertical image boundaries and window scales (of 64 x 64) to be used.

Due to the fact that vehicles are smaller on camera images when farther away, smaller scales of search windows 
are more relevant on the horizon than larger scales. Due to that I limited the lower y boundary.
Above the horizon no vehicles are expected so no search windows are applied at the top half of the image.

I chose as scales 1, 1.5, 2 and 3 to detect vehicles in different sizes in the camera images.
The maximum scale of 3 is chosen because 192 x 192 is the maximum area covered by a very close vehicle in the videos.
I am using a horizontal and vertical overlap of 75% between sliding windows in order to detect vehicles at various 
positions on an image. 75% is a good tradeoff between resulting computation performance impact (number of windows)
and accuracy on vehicle detections.

The following images show the windows that where used. Each image is showing a different scale.
The colors green and blue are used so the overlapping rectangles can be distinguished. 
![alt text][windows_0]
![alt text][windows_1]
![alt text][windows_2]
![alt text][windows_3]
![alt text][windows_4]
![alt text][windows_5]
![alt text][windows_6]
![alt text][windows_7]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

To optimize the performance of my classifier, I chose to adopt the extraction of hog features once per image 
and window scale since the computation performance of hog feature extraction per sliding window was too low 
to tune the pipeline. I also reduced the number of search windows by using proper limits on upper and lower y value.

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
Each bounding box for a vehicle detection added 1 to the heat map. I applied a threshold  on this heatmap 
to derive the final vehicle detection bounding boxes. 
I also used a history of the heatmaps of the last frames to filter our false positives. I combined (added) the
heatmaps of the last frames and the current frame. This should help to distinguish fast moving static objects 
(moving fast in the camera images) from slow moving other vehicle that drive in the same direction as the 
ego vehicle. This filters most false positives in static objects like the road surface.
The code for this part is in section 'Filtering False Positives using Heatmap' in the jupyter notebook
and in the function 'process_video_image'.

Here is the heatmap for detected bounding boxes in test image 1:
![alt text][heatmap]


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I tried to choose feature extraction parameters (mainly for HOG) based on the classifier accuracy, feature extraction
computation performance and low feature number. I finally used all channels of YCrCb color space which lead to 
a comparably large feature vector (4800 features). Before I was trying to tune the pipeline with a classifier 
using only the Y channel of this same color space but there were too many false positives on test images.

The classifier I used has a high test data set accuracy. But the performance in real videos appeared much worse.
The heatmap approach helped to remove most of the false positives but not all. To get a more trustful evaluation
result for the classifier, I could manually filter the images used for the test data set by removing all but one image
from the nearly same perspective on the same car.

The training data set contains mostly only images from the back of vehicles. Thus crossing vehicles which are 
seen from the side and incoming vehicles might not be detected. In addition, my approach on using the heatmaps
from previous frames is best suited to detect vehicle moving with a low relative velocity in the same direction 
in front of the ego vehicle. This could be improved by adding more training images with vehicles from other angles.
Also the bounding boxes detected in a video for the same car differs (wobbles) over time (framewise). This could be
improved by detecting the connected vehicle pixels inside the bounding boxes coming from the labeled heatmap.
Then a much more precise bounding box could be determined around the vehicle pixels.
Also, the bounding box for the same car could be smoothed over tme.
I also could introduce adding search windows on the region where a vehicle has been detected in the previous frame.
