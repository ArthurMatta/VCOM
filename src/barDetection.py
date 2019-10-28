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
    img = cv.imread(filename)
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
    Filter the image to remove its noise by blurring and thresholding the image.

    :param img:
    :return filtered image:
    """

    blurred = cv.blur(img, (9, 9))
    (_, thresh) = cv.threshold(blurred, 225, 255, cv.THRESH_BINARY)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (35, 7))
    closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    closed = cv.erode(closed, None, iterations=4)
    closed = cv.dilate(closed, None, iterations=4)
    return closed


def find_contours(img):
    """
    Find the contours in a thresholded image.

    :param img:
    :return contours:
    """

    contours = cv.findContours(img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = iu.grab_contours(contours)

    if len(contours) == 0:
        return None

    c = sorted(contours, key=cv.contourArea, reverse=True)[0]
    rect = cv.minAreaRect(c)
    box = cv.boxPoints(rect)
    box = np.int0(box)

    return box


def mask_image(coordinates, image):
    """
    Apply a mask in the image to isolate the barcode from the background.

    :param coordinates:
    :param image:
    :return masked image:
    """

    mask = np.zeros(image.shape, dtype=np.uint8)
    roi = np.array([coordinates], dtype=np.int32)
    channel_count = image.shape[2]
    ignore_mask_color = (255,) * channel_count
    cv.fillPoly(mask, roi, ignore_mask_color)
    masked = cv.bitwise_and(image, mask)

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
    """
    Crop the barcode from the image.

    :param img:
    :param coordinates:
    :return cropped barcode:
    """

    max_x, min_x, max_y, min_y = find_extremes(coordinates)
    img = img[min_y:max_y, min_x:max_x]
    return img


def stretch(img, coordinates):
    """
    Rearrange the barcode in order for it to be displayed as in frontal view.

    :param img:
    :param coordinates:
    :return:
    """

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


def bar_detection(image):
    # Convert image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Highlight details in the image
    gradient = compute_gradient(gray)
    # Further highlight and blur the rest of the image
    closed = filter_image(gradient)
    # Find the box that contains the barcode
    box = find_contours(closed)

    return box


def image_detection(filename):
    # Open image file
    image = openfile(filename)
    # Detect barcode
    box = bar_detection(image)

    if box is not None:
        # Attempt image correction
        masked_image = mask_image(box, image)
        # Crop and stretch image
        final = crop(masked_image, box)
        # Warp barcode to window
        warp = stretch(masked_image, box)

        cv.imshow("Final", final)
        cv.imshow("Warp", warp)
    else:
        cv.imshow("Image", image)

    cv.waitKey(0)
    cv.destroyAllWindows()


def video_detection():
    cap = cv.VideoCapture(0)

    while True:
        _, frame = cap.read()
        frame = cv.flip(frame, 1)

        box = bar_detection(frame)

        if box is not None:
            cv.drawContours(frame, [box], -1, (0, 255, 0), 3)

        cv.imshow("Frame", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv.destroyAllWindows()
            break


if __name__ == '__main__':
    video_detection()
