
# CarND-Advanced-Lane-Lines
##Advanced Lane Finding Project

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

[image1]: ./output_images/chessboard_calibration.png "Undistorted"
[image2]: ./output_images/test_calibration.png "Road Undistorted"
[image3]: ./output_images/binary.png "Binary Example"
[image4]: ./output_images/box_lane_lines.png "Box Lane Lines"
[image5]: ./output_images/straight_lane_lines.png "Warp Example"
[image6]: ./output_images/histogram.png "Histogram"
[image7]: ./output_images/find_windows.png "Fit Windows"
[image8]: ./output_images/find_previous.png "Fit Previous"
[image9]: ./output_images/project_lane_lines.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the code cell titled "Camera Calibration" of the IPython notebook located in `./p4.ipynb`.

```python
def calibrate_camera(image_path):
    image_names = glob.glob(image_path)
    # image_names = ["camera_cal/calibration2.jpg"]

    obj_points = []
    img_points = []

    # Object Points are in same z plane, so all z coordinates are 0.
    # Obj_p are corners: (0,0,0), (1,0,0), ..., (7,5,0)
    obj_p = np.zeros((9*6,3), np.float32)
    obj_p[:,0:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    for img_name in image_names:
        img = plt.imread(img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret_val, corners = cv2.findChessboardCorners(gray, (9,6), None)
        # If corners found, append the corners to img_points, append obj_p to object_points.
        if ret_val:
            img_points.append(corners)
            obj_points.append(obj_p)

            # draw_img = cv2.drawChessboardCorners(img, (9,6), corners, ret_val)
            # plt.imshow(draw_img)
            # plt.show()
    return cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1],None,None)
```

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `obj_p` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  

Using the output of `cv2.calibrateCamera()`, I applied this distortion correction to the test image using the `cv2.undistort()` function.  That code can be found in the code block titled "Undistort":

```python
ret, mtx, dist, rvecs, tvecs = calibrate_camera("camera_cal/calibration*.jpg")

def undistort(img):
    return cv2.undistort(img, mtx, dist, None, mtx)
```

Using this, I obtained this result:

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
Similar to undistorting the chessboard image above, I simply called undistort on the test image to get an undistorted image of the road, resulting in the following:
![alt text][image2]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
The code for this step is contained in the code cell titled "Create Binary Images" of `./p4.ipynb`.

I used a combination of methods to create a thresholded binary:
 
 1. Combination of Sobel X & Y gradients (Red)
 2. Combination of H & S channel from HLS colorspace to find yellow (Green)
 3. Combination of S & V channel from HLS & HSV colorspaces to find bright whites (Blue)

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform(img)`, which appears in the code blocked titled "Bird's Eye Transform" in `p4.ipynb`.  The `perspective_transform(img)` function takes as input an image (`img`). Inside the method, the src and dst corners are calculated, and used to calculate the M and M_inv matrices.  The image and M matrix are then passed to the function `warp(img, M)`, which returns the image warped to a birds eye view.  

```python
# Returns warp perspective of image using M as matrix.
def warp(img, M):
    img_shape = (img.shape[1],img.shape[0])
    return cv2.warpPerspective(img, M, img_shape, flags=cv2.INTER_LINEAR)

# Calculates source/destination points for perspective transform, and warps image to birds eye
def perspective_transform(img):
    # Define parameters for calculating source and destination points
    width, height = img.shape[1],img.shape[0]
    bottom_width_perc = 0.74
    top_width_perc = 0.12
    height_perc = 0.64
    bottom_perc = 0.935
    top_y = height*height_perc
    bottom_y = height*bottom_perc
    top_x_offset = width*top_width_perc/2
    bottom_x_offset = width*bottom_width_perc/2
    midpoint = width * 0.5
    offset = width * 0.2
    
    # Define source corners
    src_corners = np.float32([[midpoint-top_x_offset,top_y],
                              [midpoint+top_x_offset,top_y],
                              [midpoint+bottom_x_offset,bottom_y],
                              [midpoint-bottom_x_offset,bottom_y]])
    
    # Define destination corners
    dst_corners = np.float32([[offset,0],[width-offset,0],[width-offset,height],[offset,height]])
    M = cv2.getPerspectiveTransform(src_corners, dst_corners)
    M_inv = cv2.getPerspectiveTransform(dst_corners, src_corners)
    warped = warp(img, M)
    return warped, M, M_inv
```
The source points resulted in the following images:

![alt text][image4]

I verified that my perspective transform was working as expected by viewing the warped image of the `test_images/straight_lines1.jpg` file, and verifying that the lane lines were vertical:

![alt text][image5]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To find the lane lines, I created an abstract base class `LaneLine` with subclasses `LeftLaneLine` and `RightLaneLine`, as well as another class called `LaneFinder`.  The code for these classes can be found in the code block titled "Finding Lane Lines" in `p4.ipynb`.

`LaneFinder` managed an instance of `LeftLaneLine` and `RightLaneLine`, managing finding the lane lines for a given image as well as calculating the average radius of curvature and distance from the center of the lane.

`LaneLine` contained most of the shared code between `LeftLaneLine` and `RightLaneLine`, except for finding the base of the lane position using a histogram (`LeftLaneLine` searched the left half of the histogram for the peak, `RightLaneLine` searched the right half of the histogram for the peak).

![alt text][image6]

`LaneFinder` calls `find_lane_line(bin_warped_img)` for each lane line, for every frame of the video.  This method does the following:

1. Find current fit:
  * If line not previously detected: use sliding windows to find lane line
  * Else: Use previous line to search for lane line within margin
2. Set the current fit and previous fit to the class variables
3. Calculate x,y fit pixels for the lane line using the current fit
4. Calculate the average x,y fit pixels across the last 15 frames (best_x_fitted)
5. Calculate the base position of the lane line (x coordinate of fit at bottom of image)
6. Calculate radius of curvature for the best_x_fitted average fit.

If the lane lines had not been previously detected, a sliding window approach was taken to finding the line.  The code for this is in the method `find_current_fit_with_windows(bin_warped_img)` in the `LaneLine` class. 
This method did the following:

1. Get all nonzero pixel coordinates (x and y)
2. Get base location of lane line using histogram
3. Calculate window from the base location, and find all nonzero pixels inside the window
4. Append found nonzero pixels to list
4. If the number of nonzero pixels found is greater than a minpix threshold, set the new base location of the next window to the mean of the previously found nonzero pixels.
5. Loop, moving window up from bottom to top of image
6. Use appended list of all found nonzero pixels to calculate polynomial fit, and return fit

This is the result:

![alt text][image7]

If the lane lines had been previously detected, we looked for the lane line within a margin of the previously found lane line.  The code for this is in the method `find_current_fit_from_previous(bin_warped_img)` in the `LaneLine` class. 
This method did the following:

1. Get all nonzero pixel coordinates (x and y).
2. Find all nonzero pixels that lie within a margin of the previous lane line (best_x_fitted).
4. If the number of nonzero pixels found is less than a minpix threshold, then use the instead use the sliding window approach.
5. Else, calculate a polynomial fit from the found nonzero pixels within the margin and return the fit.

This is the result:

![alt text][image8]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

`LaneFinder` calculates the average radius of curvature and distance from the center of the lane.  These can be found in the methods `get_distance_from_center(image_width)` and `get_curve_radius()`.  

For calculating the radius of curvature, `get_curve_radius()` of `LeftLaneLine` and `RightLaneLine` to retrieve the individual radius of curvatures in the field `radius_of_curvature`.  It then averages these values and returns it.

For calculating the distance from the center, `get_distance_from_center(image_width)` takes in the width of the image in pixels.  It then calculates the camera center location by averaging the line base position of `LeftLaneLine` and `RightLaneLine`.  It then subtracts the center of the image (image_width/2) from the camera center, and multiplies that by a constant `xm_per_pix` which translates pixels to meters in the x axis.  A negative value indicates the car is right of center, and a positive value indicates the car is left of center.


####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the code block "Project Lines on Original Image" in `p4.ipynb`.  Here is the code:

```python
# Returns lane lines (filled) projected on to the unwarped (non birds eye) view.
def project_lines_on_unwarped(unwarped_img, bin_warped_img, left_lane_line, right_lane_line, M_inv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(bin_warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, bin_warped_img.shape[0]-1, bin_warped_img.shape[0])
    pts_left = np.array([np.transpose(np.vstack([left_lane_line.best_x_fitted, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane_line.best_x_fitted, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = warp(color_warp, M_inv)
    # Combine the result with the original image
    result = cv2.addWeighted(unwarped_img, 1, newwarp, 0.3, 0)
    return result
```

This method creates a separate image to draw the lane lines (filled in green) and then overlays that image onto the unwarped original image (still undistorted).  It uses the `M_inv` inverse warp matrix previously calculated in `perspective_transform(img)`.  It also takes in the left and right lane lines, and uses the `best_x_fitted` (fitted line averaged over the last 15 frames) to draw the boundaries of the lane lines.  Those lane lines are then unwarped using M_inv, and added to the original unwarped image.  The result being:

![alt text][image9]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had a hard time determining sanity checks in my lane finding that would indicate that I needed to switch from using the previous lane lines to find the current lane line to using the sliding window approach.  I didn't end up needing it to successfully complete the project video, but I definitely will to complete the challenge video.  

The challenge video is quite challenging.  The camera perspective appears to be different than the project video, which means that the area warped to birds eye is different (larger in the challenge video).  This means I'm looking at a larger area, introducing more possibilities of finding non lane lines in my binary image.  The highway wall in particular is setting off my gradients, which is completely skewing my lane finding.  However, if I change the source/destination points in the perspective shift, it could negatively affect my ability to perform on the project video.  

I'd like to come back to this after I complete project 5, and attempt to tune my binary thresholds as well as my perspective source/destination points and work on completing the challenge video.