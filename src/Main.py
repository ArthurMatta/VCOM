import cv2 as cv
import imutils as iu
import numpy as np

filepath = "TestFiles/"


def openfile(filename):
    """
    Load the image.

    :param filename:
    :return grayscale image;
    """

    global filepath
    img = cv.imread(f'{filepath}{filename}')
    height, width, _ = img.shape
    print(f'Initial image Width: {width}, Height: {height}')
    return img


def compute_gradient(img):
    """
    Construct the gradient magnitude representation of the grayscale image
    in the horizontal and vertical directions.

    :param img:
    :return gradient;
    """

    grad_x = cv.Sobel(img, cv.CV_32F, dx=1, dy=0, ksize=-1)
    grad_y = cv.Sobel(img, cv.CV_32F, dx=0, dy=1, ksize=-1)

    gradient = cv.subtract(grad_x, grad_y)
    gradient = cv.convertScaleAbs(gradient)

    return gradient


def filter_image(img):
    """
    Filter the image to remove its noise.

    :param img:
    :return filtered image:
    """

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (21, 7))
    img = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    img = cv.erode(img, None, iterations=4)
    img = cv.dilate(img, None, iterations=4)

    return img


def find_contours(img):
    """
    Find the contours of the barcode.

    :param img:
    :return contours:
    """

    contour = cv.findContours(img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = iu.grab_contours(contour)
    c = sorted(contour, key=cv.contourArea, reverse=True)[0]

    rect = cv.minAreaRect(c)
    contours = cv.boxPoints(rect)
    contours = np.int0(contours)

    print(f'Boxed contents area: \n{contours}')
    print("Barcode found at:")
    print(f'y1:{contours[2][1]}, y2:{contours[0][1]}, x1:{contours[0][0]}, x2:{contours[2][0]}')

    return contours


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


def main():
    print("Barcode Reader Start\n")

    filename = "026245406421_2.jpg"

    # Open image file
    image = openfile(filename)
    # Convert image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Highlight details in the image
    gradient = compute_gradient(gray)
    # Further highlight and blur the rest of the image
    closed = filter_image(gradient)
    # Find the box that contains the barcode
    contours = find_contours(closed)
    # Attempt image correction
    masked_image = mask_image(contours, filename)
    # Crop and stretch image
    final = crop(masked_image, contours)
    # Warp barcode to window
    warp = stretch(masked_image, contours)

    cv.imshow('Final', final)
    cv.imshow('Warp', warp)

    cv.waitKey(0)


main()
