from collections import deque
from helper import *
import cv2
import numpy as np

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    """
    if calc_mean:
        assert('left_mem' in kwargs.keys())
        assert('right_mem' in kwargs.keys())
    """
    original_img = np.copy(image)
    
    # convert to grayscale
    gray_img = grayscale(image)
    
    # darken the grayscale
    darkened_img = adjust_gamma(gray_img, 0.5)
    
    # Color Selection Selected hls space because was getting error with HSV while parsing videos. Bcz for some frames it was not detecting anything
    white_mask = isolate_color_mask(to_hls(image), np.array([0, 200, 0], dtype=np.uint8), np.array([200, 255, 255], dtype=np.uint8))
    yellow_mask = isolate_color_mask(to_hls(image), np.array([10, 0, 100], dtype=np.uint8), np.array([40, 255, 255], dtype=np.uint8))
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    colored_img = cv2.bitwise_and(darkened_img, darkened_img, mask=mask)
    
    # Apply Gaussian Blur
    blurred_img = gaussian_blur(colored_img, kernel_size=7)
    
    # Apply Canny edge filter
    canny_img = canny(blurred_img, low_threshold=70, high_threshold=140)
    
    # Get Area of Interest
    aoi_img = get_aoi(canny_img)
    
    # Apply Hough lines
    rho=1
    theta=np.pi/180
    threshold=20
    min_line_len=20
    max_line_gap=300
    hough_lines = cv2.HoughLinesP(aoi_img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    hough_img = draw_lines(original_img, hough_lines)
    
    # Extrapolation and averaging
    left_lane, right_lane = get_lane_lines(original_img, hough_lines)
    
    result = draw_weighted_lines(original_img, [left_lane, right_lane], thickness= 10)
       
    return result