## Writeup Template

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_calibration1.jpg "Undistorted"
[image2]: ./output_images/undistort_straight_lines1.jpg "Road Transformed"
[image3]: ./output_images/binary_straight_lines1.jpg "Binary Example"
[image4]: ./output_images/perspective_transform_straight_lines1.jpg "Warp Example"
[image5]: ./output_images/poly_straight_lines1.jpg "Fit Visual"
[image6]: ./output_images/test1.jpg "Output"
[video1]: ./output_video/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. I used `objp` and `corners` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color thresholds to generate a binary image (thresholding steps at lines 26 through 50 in `Main.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears in lines 53 through 58 in the file `Main.py`.  The `perspective_transform()` function takes as inputs an image (`img`), and uses source (`src_points`) and destination (`dst_points`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src_points = np.float32([[576, 464], [710, 464], [1045, 677], [267, 677]])
dst_points = np.float32([[200, 0], [1000, 0], [1000, 700], [200, 700]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 576, 464      | 200, 0        | 
| 710, 464      | 1000, 0       |
| 1045, 677     | 1000, 700     |
| 267, 677      | 200, 700      |

Here is the result of the trasform.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 136 through 143 in my code in `Main.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 194 through 207 in my code in `Main.py` in the function `createBlankImageWithLanes(shape, left_fitx, right_fitx, ploty, M)`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I used the most of the code that was already presented in project, but the default settings didn't work quite well with challenge_video. Many extra edges created a lot of problem for recognition, so I decided to focus on thresholding. I deleted the sobel binary and used only color spaces. I've chosen one binary for white line on the video (red channel, RGB) and other binary for yellow line (hue and saturation channels of HSL) and made joint binary that pretty decently helps to recognize the both lanes on challenged video. But this is not enough for harder_challenged_video. For this video I think make sence to add the additional sanity checks and maybe use the information as distance from center on the "finding lane" phase of the algorithm. 
