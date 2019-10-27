import cv2 as cv
import imutils as iu
import numpy as np
import sys

filepath = "TestFiles/"


def openfile(filename):
    global filepath
    img = cv.imread(f'{filepath}{filename}', 0)
    height, width = img.shape
    print(f'Initial image Width: {width}, Height: {height}')
    return img


def highlight_details(img):
    sobel_x = cv.Sobel(img, cv.CV_32F, dx=1, dy=0, ksize=-1)
    sobel_y = cv.Sobel(img, cv.CV_32F, dx=0, dy=1, ksize=-1)

    img = cv.subtract(sobel_x, sobel_y)
    img = cv.convertScaleAbs(img)

    return img


def highlight_code(img):
    blurred = cv.blur(img, (9, 9))
    (_, thresh) = cv.threshold(blurred, 225, 255, cv.THRESH_BINARY)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (21, 7))
    img = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    img = cv.erode(img, None, iterations=4)
    img = cv.dilate(img, None, iterations=4)

    return img


def find_box(img):
    contour = cv.findContours(img.copy(), cv.RETR_EXTERNAL,
                           cv.CHAIN_APPROX_SIMPLE)
    contour = iu.grab_contours(contour)
    c = sorted(contour, key=cv.contourArea, reverse=True)[0]

    rect = cv.minAreaRect(c)
    box = cv.boxPoints(rect)
    box = np.int0(box)

    print(f'Boxed contents area: \n{box}')
    print("Barcode found at:")
    print(f'y1:{box[2][1]}, y2:{box[0][1]}, x1:{box[0][0]}, x2:{box[2][0]}')

    return box


def mask_image(coordinates, filename):
    global filepath
    img = cv.imread(f'{filepath}{filename}', -1)
    mask = np.zeros(img.shape, dtype=np.uint8)
    roi = np.array([coordinates], dtype=np.int32)
    channel_count = img.shape[2]
    ignore_mask_color = (255,) * channel_count
    cv.fillPoly(mask, roi, ignore_mask_color)
    masked = cv.bitwise_and(img, mask)

    return masked


def find_extremes(coordinates):
    max_x, min_x, max_y, min_y = 0, 10000, 0, 10000
    for dot in coordinates:
        if max_x < dot[0]: max_x = dot[0]
        if min_x > dot[0]: min_x = dot[0]
        if max_y < dot[1]: max_y = dot[1]
        if min_y > dot[1]: min_y = dot[1]

    return max_x, min_x, max_y, min_y


def crop(img, coordinates):
    max_x, min_x, max_y, min_y = find_extremes(coordinates)
    img = img[min_y:max_y, min_x:max_x]
    return img


def stretch(img, coordinates):
    # reshape borders
    pts = coordinates.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # compute the difference between the points -- the top-right
    # will have the minimum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # multiply the rectangle by the original ratio
    rect *= 1

    # now that we have our rectangle of points, let's compute
    # the width of our new image
    (tl, tr, br, bl) = rect
    width_a = width_b = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))

    # ...and now for the height of our new image
    height_a = height_b = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    max_width = max(int(width_a), int(width_b))
    max_height = max(int(height_a), int(height_b))

    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv.getPerspectiveTransform(rect, dst)
    warp = cv.warpPerspective(img, M, (max_width, max_height))

    return warp

def dotted(img):

    # Define Dotted Colors
    red = [0,0,255]
    blue = [255,0,0]

    # Get Shape
    height, width, channels = img.shape

    # Get Middle Height
    middleHeight = int(height/2)

    # Get 1/6 Height
    sixHeight = int(height/6)

    # Plot Dotted Line for Barcode
    for x in range(width):
        # Get Color
        pixel = img[middleHeight, x]

        # Gray Value
        gray = 0.02126 * pixel[2] + 0.7152 * pixel[1] + 0.0722 * pixel[0]

        # Decide Result Based On Middle Gray
        final = blue
        if gray < 125:
            final = red

        # Apply Result in 2/6th of Height
        for y in range(sixHeight*2):
            img[middleHeight + y - sixHeight, x] = final

    # Show Result
    cv.imshow('Dotted', img)

def main():
    print("Barcode Reader Start\n")

    filename = "026245406421_2.jpg"

    # Use argument file name
    if len(sys.argv) == 2:
        filename = sys.argv[1]

    # Open image file
    image = openfile(filename)
    # Highlight details in the image
    gradient = highlight_details(image)
    # Further highlight and blur the rest of the image
    closed = highlight_code(gradient)
    # Find the box that contains the barcode
    box = find_box(closed)
    # Attempt image correction
    masked_image = mask_image(box, filename)
    # Crop and stretch image
    final = crop(masked_image, box)
    # Warp barcode to window
    warp = stretch(masked_image, box)

    # Show final detection
    cv.imshow('Final', final)
    cv.imshow('Warp', warp)

    # Plot Barcode
    dotted(final)

    cv.waitKey(0)

main()
