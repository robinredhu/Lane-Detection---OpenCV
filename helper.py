import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import os

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            
    return img

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def to_hls(img):
    """
        Converting to HLS Color Space
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

def to_hsv(img):
    """
        Converting to HSV Color Space
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

def isolate_color_mask(img, low_threshold, high_threshold):
    """
        Selecting pixels that lies in the given threshold
    """
    return cv2.inRange(img, low_threshold, high_threshold)

def adjust_gamma(image, gamma=1.0):
    """
        Darkening the image to better highlight the White and Yellow strips
        For reference please visit "https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/"
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def save_imgs(image_list, labels, prefix="Test", folder="test_imgs_output"):
    """
        Saving output folder
    """
    # Checking if path exists or not. If not it will create a new folder
    if not os.path.exists(folder):
        os.mkdir(folder)
    # Saving output images to the output folder
    for i in range(len(image_list)):
        PATH = folder + "/" + prefix + "_" + labels[i]
        Image.fromarray(image_list[i]).save(PATH)

def display_imgs(img_list, labels=[],cols=2, fig_size=(15,15)):
    """
        PLoting all images
    """
    cmap = None
    tot = len(img_list)
    rows = tot / cols
    plt.figure(figsize=fig_size)
    for i in range(tot):
        plt.subplot(rows, cols, i+1)
        if len(img_list[i].shape) == 2:
            cmap = 'gray'
        if len(labels) > 0:
            plt.title(labels[i])
        plt.imshow(img_list[i], cmap=cmap)
        
    plt.tight_layout()
    plt.show()

def get_aoi(img):
    """
        Selecting area of interest to be looked upon
    """
    mask = np.zeros_like(img)
    top_left = [img.shape[1] * 0.4, img.shape[0] * 0.6]
    top_right = [img.shape[1] * 0.6, img.shape[0] * 0.6]
    bottom_left = [img.shape[1] * 0.1, img.shape[0]]
    bottom_right = [img.shape[1] * 0.95, img.shape[0]]
    
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255, ) * mask.shape[2])  # i.e. 3 or 4 depending on your image
    return cv2.bitwise_and(img, mask)

def get_hough_lines(img, rho=1, theta=np.pi/180, threshold=20, min_line_len=20, max_line_gap=300):
    """
    `img` should be the output of a Canny transform.
        
    Returns hough lines
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), 
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def calculate_slope_intercept(line):
    """
        Calculating slope nd intercept for a line
    """
    for x1, y1, x2, y2 in line:
        if x2-x1 == 0:
            return math.inf, 0
    slope = (y2-y1)/(x2-x1)
    intercept = y1 - slope * x1
    return slope, intercept
        
def get_lines_slope_intecept(lines):
    """
        Calculating slope and intercept for all lines and then taking average of left side lines and right side lines
        If slope is negativce that means it is left lane and vice versa
    """
    left_lines = []
    right_lines = []
    for line in lines:
        slope, intercept = calculate_slope_intercept(line)
        if slope == math.inf:
            continue
        if slope < 0:
            left_lines.append((slope, intercept))
        else :
            right_lines.append((slope, intercept))

    # Calculating average slope and avg intercept for left lane
    slope = intercept = 0
    for line in left_lines:
        slope += line[0]
        intercept += line[1]
    left_avg = [slope/len(left_lines), intercept/len(left_lines)]
    
    # Calculating average slope and avg intercept for right lane
    slope = intercept = 0
    for line in right_lines:
        slope += line[0]
        intercept += line[1]
    right_avg = [slope/len(right_lines), intercept/len(right_lines)]
    
    return left_avg, right_avg

def convert_slope_intercept_to_line(y1, y2 , line):
    """
        Fetching end coordinates from line from equation :  y = mx + c
    """
    if line is None:
        return None
    
    slope, intercept = line
    x1 = int((y1- intercept)/slope)
    y1 = int(y1)
    x2 = int((y2- intercept)/slope)
    y2 = int(y2)
    return((x1, y1),(x2, y2))

def get_lane_lines(img, lines):
    """
        This function will generate the left and right lanes
    """
    left_avg, right_avg = get_lines_slope_intecept(lines)
    
    y1 = img.shape[0]
    y2 = img.shape[0] * 0.6
    
    left_lane = convert_slope_intercept_to_line(y1, y2, left_avg)
    right_lane = convert_slope_intercept_to_line(y1, y2, right_avg)
    return left_lane, right_lane

def draw_weighted_lines(img, lines, color=[255, 0, 0], thickness=2, alpha = 1.0, beta = 0.95, gamma= 0):
    mask_img = np.zeros_like(img)
    for line in lines:
        if line is not None:
            cv2.line(mask_img, *line, color, thickness)            
    return weighted_img(mask_img, img, alpha, beta, gamma)