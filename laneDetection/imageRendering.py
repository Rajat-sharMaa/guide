import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

#---------------------------------------------------------------------------------------
#Function for cropping the region of interest

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255  # <-- This line altered for grayscale.

    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#---------------------------------------------------------------------------------------

#Function for rendering lines on image

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
# If there are no lines to draw, exit.
        if lines is None:
            return
# Make a copy of the original image
        img = np.copy(img)
# Create a blank image that matches the original in size.
        line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
        )
# Loop over all lines and draw them on the blank image.
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
# Merge the image with the lines onto the original.
        img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
# Return the modified image.
        return img


#---------------------------------------------------------------------------------------

#Reading the test image

image = mpimg.imread('test_images/solidWhiteCurve.jpg')
height = image.shape[0]
width = image.shape[1]


region_of_interest_vertices = [
    (0, height),
    (width / 2, height / 2),
    (width, height),
]
plt.figure()
plt.imshow(image)
plt.title("Orignal image")
plt.show()
#---------------------------------------------------------------------------------------

#Converting to grayscale

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(gray_image)
plt.title("Grayscale")
plt.show()
#---------------------------------------------------------------------------------------

#Canny edge detection


cannyed_image = cv2.Canny(gray_image, 100, 200)
plt.imshow(cannyed_image)
plt.title("Edge Detection")
plt.show()

#---------------------------------------------------------------------------------------

#Cropping out the region of interest

cropped_image = region_of_interest(
    cannyed_image,
    np.array(
        [region_of_interest_vertices],
        np.int32
    ),
)

plt.imshow(cropped_image)
plt.title("Cropped Region of Interest")
plt.show()


#---------------------------------------------------------------------------------------

# Using Hough Transforms to Detect Lines

lines = cv2.HoughLinesP(
    cropped_image,
    rho=6,
    theta=np.pi / 60,
    threshold=160,
    lines=np.array([]),
    minLineLength=40,
    maxLineGap=25
)
print(lines)

#---------------------------------------------------------------------------------------

#Rendering Detected Hough Lines as an Overlay

line_image = draw_lines(image, lines)
plt.figure()
plt.imshow(line_image)
plt.title("Rendered Lines")
plt.show()

#---------------------------------------------------------------------------------------

#Creating a Single Left and Right Lane Line

left_line_x = []
left_line_y = []
right_line_x = []
right_line_y = []
for line in lines:
    for x1, y1, x2, y2 in line:
        slope = (y2 - y1) / (x2 - x1)  # <-- Calculating the slope.
        if math.fabs(slope) < 0.5:  # <-- Only consider extreme slope
            continue
        if slope <= 0:  # <-- If the slope is negative, left group.
            left_line_x.extend([x1, x2])
            left_line_y.extend([y1, y2])
        else:  # <-- Otherwise, right group.
            right_line_x.extend([x1, x2])
            right_line_y.extend([y1, y2])

#---------------------------------------------------------------------------------------

#Creating a Single Linear Representation of each Line Group

min_y = int(image.shape[0] * (3 / 5))  # <-- Just below the horizon
max_y = int(image.shape[0])  # <-- The bottom of the image


poly_left = np.poly1d(np.polyfit(
    left_line_y,
    left_line_x,
    deg=1
))

left_x_start = int(poly_left(max_y))
left_x_end = int(poly_left(min_y))

poly_right = np.poly1d(np.polyfit(
    right_line_y,
    right_line_x,
    deg=1
))


right_x_start = int(poly_right(max_y))
right_x_end = int(poly_right(min_y))

line_image = draw_lines(
    image,
    [[
        [left_x_start, max_y, left_x_end, min_y],
        [right_x_start, max_y, right_x_end, min_y],
    ]],
    thickness=5,
)
plt.figure()
plt.imshow(line_image)
plt.title("Single line represetation of each line group")
plt.show()
