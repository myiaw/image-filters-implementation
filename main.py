import cv2
import numpy as np
import cv2 as cv


def my_roberts(input_image):
    # We have two kernels, one for each axis.
    kernel_horizontal = np.array([[1, 0], [0, -1]])
    kernel_vertical = np.array([[0, 1], [-1, 0]])

    # Get only height and width from shape of input_image.
    height, width = input_image.shape[:2]

    # Create empty image, so we can add edges onto it.
    result_image = np.zeros((height, width), dtype=np.uint8)

    # Loop through the entire image.
    for i in range(0, height - 1):
        for j in range(0, width - 1):
            # Kernel is 2x2, so we loop by 4 pixels -> 2 vertical, 2 horizontal
            window = input_image[i:i + 2, j:j + 2]

            # Multiply each kernel with the location matrix.
            x_conv = np.sum(np.multiply(window, kernel_horizontal))
            y_conv = np.sum(np.multiply(window, kernel_vertical))

            # Add them together.
            magnitude = np.sqrt(x_conv ** 2 + y_conv ** 2)
            # Put the magnitude at the current pixel
            result_image[i, j] = magnitude

    return result_image


def my_sobel(input_image):
    # We have two kernels, one for each axis.
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Get only height and width from shape of input_image.
    height, width = input_image.shape[:2]

    # Create empty image, so we can add edges onto it.
    result_image = np.zeros((height, width), dtype=np.uint8)

    # Loop through the entire image.
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Kernel is 3x3, so we loop by 9 pixels -> 3 vertical, 3 horizontal
            window = input_image[i - 1:i + 2, j - 1:j + 2]

            # Multiply each kernel with the location matrix.

            x_conv = np.sum(np.multiply(window, kernel_x))
            y_conv = np.sum(np.multiply(window, kernel_y))

            # Add them together.
            magnitude = np.sqrt(x_conv ** 2 + y_conv ** 2)
            result_image[i, j] = magnitude

    return result_image


def my_prewitt(input_image):
    # We have two kernels, one for each axis.
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    # Get only height and width from shape of input_image.
    height, width = input_image.shape[:2]

    # Create empty image, so we can add edges onto it.
    result_image = np.zeros((height, width), dtype=np.uint8)

    # Loop through the entire image.
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Kernel is 3x3, so we loop by 9 pixels -> 3 vertical, 3 horizontal
            roi = input_image[i - 1:i + 2, j - 1:j + 2]

            # Multiply each kernel with the location matrix.

            x_conv = np.sum(np.multiply(roi, kernel_x))
            y_conv = np.sum(np.multiply(roi, kernel_y))

            # Add them together.

            magnitude = np.sqrt(x_conv ** 2 + y_conv ** 2)
            result_image[i, j] = magnitude

    return result_image


def change_contrast(input_image, alpha, beta):
    changed = np.clip(alpha * input_image + beta, 0, 255).astype(np.uint8)  # g(i,j)=α∗f(i,j)+β --> i,j length width
    return changed  # alpha, beta --> contrast, brightness


def my_canny(input_image, low, high):
    return cv2.Canny(input_image, low, high)


# Get the image.
image = cv2.imread("input.jpg", 0)
# Brightening
adjustedBright = change_contrast(image, alpha=1.5, beta=30)  # beta = 30 Bright
# Blurring (smoothing)
smooth = cv.GaussianBlur(adjustedBright, (3, 3), 0)

robertsBright = my_roberts(adjustedBright)
roberts = my_roberts(image)
robertsSmoothBright = my_roberts(smooth)
cv.imshow("No Brightness Roberts", roberts)
cv.imshow("Changed contrast and brightness", robertsBright)
cv.imshow("Smooth Roberts", robertsSmoothBright)

sobel = my_sobel(image)
sobelBright = my_sobel(adjustedBright)
cv2.imshow("Edge Detection sobel", sobel)
cv.imshow("Edge Detection sobel bright", sobelBright)

prewitt = my_prewitt(image)
prewittBright = my_prewitt(adjustedBright)
cv.imshow("Edge Detection prewitt", prewitt)
cv.imshow("Edge Detection prewitt bright", prewittBright)

cannnyBright = my_canny(adjustedBright, 100, 200)
canny = my_canny(adjustedBright, 100, 200)

cv2.imshow("Edge Detection canny", canny)
cv2.imshow("Edge Detection canny bright", cannnyBright)

cv2.waitKey(0)
cv2.destroyAllWindows()
