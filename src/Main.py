import cv2 as cv
import numpy as np
import imutils as iu


def openfile(filename):
    filepath = "TestFiles/"
    img = cv.imread(filepath + filename, 0)
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


def mask_image(coordinates):
    img = cv.imread('TestFiles/026245406421_2.jpg', -1)
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
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv.getPerspectiveTransform(rect, dst)
    warp = cv.warpPerspective(img, M, (maxWidth, maxHeight))

    return warp


def main():
    print("Barcode Reader Start\n")

    # Open image file
    image = openfile("026245406421_2.jpg")
    # Highlight details in the image
    gradient = highlight_details(image)
    # Further highlight and blur the rest of the image
    closed = highlight_code(gradient)
    # Find the box that contains the barcode
    box = find_box(closed)
    # Attempt image correction
    masked_image = mask_image(box)
    # Crop and stretch image
    final = crop(masked_image, box)
    # Warp barcode to window
    warp = stretch(masked_image, box)

    cv.imshow('Final', final)
    cv.imshow('Warp', warp)

    cv.waitKey(0)


main()
