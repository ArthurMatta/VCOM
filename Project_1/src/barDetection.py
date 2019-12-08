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
    in the vertical directions.

    :param img:
    :return gradient;
    """

    grad_x = cv.Sobel(img, cv.CV_32F, dx=1, dy=0, ksize=-1)
    grad_y = cv.Sobel(img, cv.CV_32F, dx=0, dy=1, ksize=-1)

    gradient = cv.subtract(grad_x, grad_y)
    gradient = cv.convertScaleAbs(grad_x)

    return gradient


def filter_image(img):
    """
    Filter the image to remove its noise by blurring and thresholding the image. This process leaves am image with
    with white highlights representing the area corresponding to the barcode.

    :param img:
    :return filtered image:
    """

    blurred = cv.blur(img, (9, 9))
    (_, thresh) = cv.threshold(blurred, 225, 255, cv.THRESH_BINARY)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (50, 5))
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
        return None, None

    c = sorted(contours, key=cv.contourArea, reverse=True)[0]
    rect = cv.minAreaRect(c)
    box = cv.boxPoints(rect)
    box = np.int0(box)

    return box, rect[2]


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
    Xs = [i[0] for i in coordinates]
    Ys = [i[1] for i in coordinates]

    min_x = min(Xs)
    max_x = max(Xs)
    min_y = min(Ys)
    max_y = max(Ys)

    if min_x < 0: min_x = 0
    if max_x < 0: max_x = 0
    if min_y < 0: min_y = 0
    if max_y < 0: max_y = 0

    return max_x, min_x, max_y, min_y


def crop(img, coordinates, rot):
    """
    Crop the barcode from the image.

    :param img:
    :param coordinates:
    :return cropped barcode:
    """

    max_x, min_x, max_y, min_y = find_extremes(coordinates)
    img = img[min_y:max_y, min_x:max_x]

    if rot < -45:
        rot += 90
    img = iu.rotate(img, rot)

    return img


# https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape
def adjust_contrast_brightness(image, histPercent=1):
    # Convert to Grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv.calcHist([gray], [0], None, [256], [0, 256])
    histSize = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, histSize):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    histPercent *= (maximum / 100.0)
    histPercent /= 2.0

    # Locate Left Cut
    minGray = 0
    while accumulator[minGray] < histPercent:
        minGray += 1

    # Locate Right Cut
    maxGray = histSize - 1
    while accumulator[maxGray] >= (maximum - histPercent):
        maxGray -= 1

    # Calculate Alpha and Beta values
    # g(i,j) = alpha * f(i,j) + beta
    # alpha = 255 / (maxGray - minGray)
    # beta = -minGray * alpha
    alpha = 255 / (maxGray - minGray)
    beta = -minGray * alpha

    return cv.convertScaleAbs(image, alpha=alpha, beta=beta)


def dotted(img):
    # Define Dotted Colors
    red = [0, 0, 255]
    blue = [255, 0, 0]

    # Get Shape
    height, width, channels = img.shape

    # Get Middle Height
    middleHeight = int(height / 2)

    # Get 1/6 Height
    sixHeight = int(height / 6)

    # Auto Adjust Brightness and Contrast
    img = adjust_contrast_brightness(img)

    # Plot Dotted Line for Barcode
    for x in range(width):

        # Get Pixel
        pixel = img[middleHeight, x]

        # Gray Value
        gray = 0.02126 * pixel[2] + 0.7152 * pixel[1] + 0.0722 * pixel[0]

        # Decide Result Based On Middle Gray
        final = blue
        if gray < 127:
            final = red

        # Apply Result in 2/6th of Height
        for y in range(sixHeight * 2):
            img[middleHeight + y - sixHeight, x] = final

    return img


# https://cosmosmagazine.com/technology/how-do-barcodes-work
def describe(img):
    print('\nStarting Barcode Description\n')

    # Get Shape
    height, width, channels = img.shape

    # Get Middle Height
    middleHeight = int(height / 2)

    # Detect Barcode Left Guard
    lGuard = -1
    for x in range(width):

        # Get Pixel
        pixel = img[middleHeight, x]

        # Found Red(Black Bar)
        if pixel[2] == 255:
            lGuard = x
            break

    print(f'Barcode Left Guard detected at: {lGuard} pixel')

    # Detect Barcode Right Guard
    rGuard = -1
    for x in range(width):

        # Get Pixel
        pixel = img[middleHeight, width - x - 1]

        # Found Red(Black Bar)
        if pixel[2] == 255:
            rGuard = width - x - 1
            break

    print(f'Barcode Right Guard detected at: {rGuard} pixel\n')

    # Calculate Total Size
    total = rGuard - lGuard

    # Describe Bars/Spaces in Percentages
    start = -1
    name = ""
    for i in range(lGuard, rGuard + 1):

        # Next Pixel
        pixel = img[middleHeight, i]

        # Detect Sequence Break Space -> Bar
        if pixel[2] == 255 and name == "Bar":
            # Calculate Percentage
            end = i
            percent = round((end - start) / total * 100, 2)
            print(f'Space with {percent}%')

            # Restart
            start = -1
            name = ""

        # Detect Sequence Break Bar -> Space
        if pixel[0] == 255 and name == "Space":
            # Calculate Percentage
            end = i
            percent = round((end - start) / total * 100, 2)
            print(f'Bar with {percent}%')

            # Restart
            start = -1
            name = ""

        # Start new measure
        if start == -1:
            # Mark start
            start = i

            # Identify Bar/Space
            if pixel[2] == 255:
                name = "Space"
            else:
                name = "Bar"

    # If Start was up something was pending
    if start != -1:
        # Calculate Percentage
        end = width
        percent = round((end - start) / total * 100, 2)

        # Flip type
        if name == "Space":
            print(f'Bar with {percent}%')
        else:
            print(f'Space with {percent}%')


def bar_detection(image):
    # Convert image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Highlight details in the image
    gradient = compute_gradient(gray)
    # Further highlight and blur the rest of the image
    closed = filter_image(gradient)
    # Find the box that contains the barcode
    box, rot = find_contours(closed)

    return box, rot


def image_detection(filename):
    # Open image file
    image = openfile(filename)
    # Detect barcode
    box, rot = bar_detection(image)

    if box is not None:
        # Attempt image correction
        masked_image = mask_image(box, image)

        pre_crop = cv.drawContours(image, [box], -1, (0, 255, 0), 3)
        cv.imshow("Pre-Crop", pre_crop)

        # Crop and stretch image
        final = crop(masked_image, box, rot)

        cv.imshow("Final", final)

        # Plot Barcode
        dot_plot = dotted(final)
        cv.imshow('Dotted', dot_plot)

        # Describe Barcode
        describe(dot_plot)
    else:
        cv.imshow("Image", image)

    cv.waitKey(0)
    cv.destroyAllWindows()


def video_detection():
    cap = cv.VideoCapture(0)

    while True:
        _, frame = cap.read()
        frame = cv.flip(frame, 1)

        box, _ = bar_detection(frame)

        if box is not None:
            cv.drawContours(frame, [box], -1, (0, 255, 0), 3)

        cv.imshow("Frame", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv.destroyAllWindows()
            break


if __name__ == '__main__':
    video_detection()
