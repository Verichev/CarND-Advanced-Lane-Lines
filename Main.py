import glob

import cv2
import numpy as np

from Line import Line

src_points = np.float32([[576, 464], [710, 464], [1045, 677], [267, 677]])
dst_points = np.float32([[200, 0], [1000, 0], [1000, 700], [200, 700]])


def calibrate_camera():
    nx = 9
    ny = 5
    img = cv2.imread('camera_cal/calibration1.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if ret == True:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], gray.shape[::-1], None, None)
        return mtx, dist
    return None


def threshold_image(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    h_channel = hls[:, :, 0]
    s_binary = np.zeros_like(s_channel)
    s_binary[(h_channel > 10) & (h_channel < 25) & (s_channel > 70) & (s_channel < 120)] = 1
    red = img[:, :, 2]
    red_binary = np.zeros_like(s_channel)
    red_binary[(red >= 200) & (red <= 255)] = 1
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Sobel x
    # sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    # abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    # scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    # sxbinary = np.zeros_like(scaled_sobel)
    # sxbinary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1

    # stacked = np.dstack((red_binary, s_binary, sxbinary * 0)) * 255
    binary = np.zeros_like(s_binary)
    binary[(red_binary == 1) | (s_binary == 1)] = 1
    # plt.imshow(sxbinary)
    # plt.show()
    return binary


def perspective_transform(img):
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    # plt.imshow(warped)
    # plt.show()
    return warped, M


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    # out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def measure_curvature_pixels(left_fit, right_fit, y_eval):

    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** (3 / 2)) / (
        2 * left_fit[0])  ## Implement the calculation of the left line here
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** (3 / 2)) / (
        2 * right_fit[0])  ## Implement the calculation of the right line here

    return left_curverad, right_curverad

def fit_polynomial(binary_img, leftx, lefty, rightx, righty):
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_img.shape[0] - 1, binary_img.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    return binary_img, left_fit, right_fit, left_fitx, right_fitx, ploty

def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

def sanityCheck(left_line, right_line):
    dist = np.mean(right_line.allx) - np.mean(left_line.allx)
    sanity = dist > 300
    return sanity

def createBlankImageWithLanes(shape, left_fitx, right_fitx, ploty, M):
    # Create an image to draw the lines on
    warp_zero = np.zeros(shape).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M, (shape[::-1]))
    return newwarp


def convertImage(img, mtx, dist, left_line=None, right_line=None):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    img = threshold_image(undist)
    binary_img, M = perspective_transform(img)
    leftx, lefty, rightx, righty = find_lane_pixels(
        binary_img) if left_line is None or right_line is None else search_around_poly(binary_img,
                                                                                       left_line.current_fit,
                                                                                       right_line.current_fit)
    if len(rightx) == 0 or len(leftx) == 0 or len(righty) == 0 or len(lefty) == 0:
        return None, None, None
    img, left_fit, right_fit, left_fitx, right_fitx, ploty = fit_polynomial(binary_img, leftx, lefty, rightx, righty)
    left_line = Line()
    left_line.current_fit = left_fit
    left_line.recent_xfitted.append(left_fitx)
    if len(left_line.recent_xfitted) > 10:
        left_line.recent_xfitted.remove(0)
    left_line.last_fits.append(left_fit)
    if len(left_line.last_fits) > 5:
        left_line.last_fits.remove(0)
    left_line.allx = left_fitx
    left_line.ally = ploty
    right_line = Line()
    right_line.current_fit = right_fit
    right_line.recent_xfitted.append(right_fitx)
    if len(right_line.recent_xfitted) > 10:
        right_line.recent_xfitted.remove(0)
    right_line.last_fits.append(right_fit)
    if len(right_line.last_fits) > 5:
        right_line.last_fits.remove(0)
    right_line.allx = right_fitx
    right_line.ally = ploty

    left_line.radius_of_curvature, right_line.radius_of_curvature = measure_curvature_pixels(left_fit, right_fit, 500)

    if not sanityCheck(left_line, right_line):
        return None, None, None
    lanes_img = createBlankImageWithLanes(binary_img.shape, np.average(left_line.recent_xfitted, 0),
                                          np.average(right_line.recent_xfitted, 0), ploty, np.linalg.inv(M))
    result = cv2.addWeighted(undist, 1, lanes_img, 0.3, 0)
    return result, left_line, right_line

def convertImages():
    mtx, dist = calibrate_camera()
    image_files = glob.glob("test_images/*.jpg")
    for file in image_files:
        img = cv2.imread(file)
        result, left_fit, right_fit = convertImage(img, mtx, dist)
        _, name = cv2.os.path.split(file)
        cv2.imwrite("output_images/%s" % name, result)

def convertVideo():
    mtx, dist = calibrate_camera()
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter('output_video/project_video.mp4', fourcc, 20.0, (1280, 720))
    cap = cv2.VideoCapture('project_video.mp4')
    left_line = None
    right_line = None
    while cap.isOpened():
        ret, img = cap.read()
        if ret == 0:
            break
        result, left_line, right_line = convertImage(img, mtx, dist, left_line, right_line)
        if result is None:
            continue
        out.write(result)
    out.release()
    cap.release()


convertImages()
