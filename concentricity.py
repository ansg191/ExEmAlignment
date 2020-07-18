import logging
import warnings

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.set_printoptions(suppress=True)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def calc_R(x, y, xc, yc):
    """Calculate distance from center of potential circle

    :param x: Real x data
    :param y: Real y data
    :param xc: Potential x center
    :param yc: Potential y center
    :type x: np.ndarray
    :type y: np.ndarray
    :type xc: float
    :type yc: float
    :return: Distance from center for each point
    :rtype: np.ndarray
    """
    return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)


def f(c, x, y):
    """Calculate residuals for least sq optimization

    :param c: (x center, y center)
    :param x: Real x data
    :param y: Real y data
    :type c: (float, float)
    :type x: np.ndarray
    :type y: np.ndarray
    :return: Residuals based on how far points are from center and radius
    :rtype: np.ndarray
    """
    ri = calc_R(x, y, *c)
    return ri - ri.mean()


def fit(x, y, center):
    """Fit circle to data points

    :param x: Real x data
    :param y: Real y data
    :param center: Possible center. This will be used as a starting point for least square fitting
    :type x: np.ndarray
    :type y: np.ndarray
    :type center: (float, float)
    :return: Tuple of residuals, center, and radius
    :rtype: (np.ndarray, tuple, float)
    """
    center, _ = optimize.leastsq(f, center, args=(x, y))
    ri = calc_R(x, y, *center)
    r = ri.mean()
    y_c = -np.sqrt(r ** 2 - (x - center[0]) ** 2) + center[1]
    return y - y_c, center, r


def fit_circle(y):
    """Fit circle to curve and find lowest point. Slowly removes outlying points to achieve better curve.

    :param y: Curve data
    :type y: np.ndarray
    :return: Tuple of min location and min value
    :rtype: np.ndarray
    """
    y_len = len(y)
    orig = y.copy()
    x = np.arange(0, len(y), dtype=np.uint16)
    center = (200, 200)
    cutoff = np.linspace(50, 2.5, 10)
    r = 0

    for i in range(10):
        diff, center, r = fit(x, y, center)
        x = x[abs(diff) < cutoff[i]]
        y = y[abs(diff) < cutoff[i]]

    y_c_orig = -np.sqrt(r ** 2 - (np.arange(0, y_len, dtype=np.uint16) - center[0]) ** 2) + center[1]

    plt.figure()
    plt.plot(np.arange(0, y_len), orig)
    plt.plot(np.arange(0, y_len), y_c_orig)

    return np.array((np.nanargmin(y_c_orig), np.nanmin(y_c_orig)))


def get_points(shape, top, bot, left, right):
    """Get the lowest points for top, bottom, left, and right curves

    :param shape: Shape of grayscale image
    :param top: Top curve
    :param bot: Bottom curve
    :param left: Left curve
    :param right: Right curve
    :type shape: (int, int)
    :type top: np.ndarray
    :type bot: np.ndarray
    :type left: np.ndarray
    :type right: np.ndarray
    :return: (top point, bottom point, left point, right point)
    :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    """
    shape = np.array(shape)
    top_point = fit_circle(top)
    bot_point = shape[::-1] - fit_circle(bot)
    left_point = fit_circle(left)[::-1]
    right_point = shape[::-1] - fit_circle(right)[::-1]
    return top_point, bot_point, left_point, right_point


# noinspection PyAugmentAssignment
def get_circle(a, b, c):
    """Calculate the Center and Radius of a circle given 3 points

    :param a: Point on circle
    :param b: Point on circle
    :param c: Point on circle
    :type a: np.ndarray
    :type b: np.ndarray
    :type c: np.ndarray
    :return: (center, radius)
    :rtype: (np.ndarray, float)
    """
    # Translate Vertex A to origin
    b = b - a
    c = c - a
    logging.debug(f"B': {b}")
    logging.debug(f"C': {c}")

    # Calculate center of translated circle
    d = 2 * (b[0] * c[1] - b[1] * c[0])
    logging.debug(f"D: {d}")
    x0 = 1 / d * (c[1] * (b[0] ** 2 + b[1] ** 2) - b[1] * (c[0] ** 2 + c[1] ** 2))
    y0 = 1 / d * (b[0] * (c[0] ** 2 + c[1] ** 2) - c[0] * (b[0] ** 2 + b[1] ** 2))
    logging.debug(f"U'x: {x0}")
    logging.debug(f"U'y: {y0}")

    # Calculate radius
    radius = np.sqrt(x0 ** 2 + y0 ** 2)

    # Translate center back to actual location
    center = np.array([x0, y0])
    center += a
    logging.info(f"Center: {center}")
    logging.info(f"Radius: {radius}")
    return center, radius


def get_center(img):
    """Gets the center and radius of outer circle in an image

    :param img: Image in BGR format
    :type img: np.ndarray
    :return: (center, radius)
    :rtype: ((float, float), float)
    """
    gray: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
    gray = cv2.bilateralFilter(gray, 5, 10, 2)
    edges: np.ndarray = cv2.Canny(gray, 0, 30)

    output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    output = cv2.add(img, output)

    cv2.imshow('edges', edges)
    cv2.imshow('output', output)

    top_edge = np.argmax(edges, axis=0)
    bot_edge = np.argmax(np.flipud(edges), axis=0)
    left_edge = np.argmax(edges, axis=1)
    right_edge = np.argmax(np.fliplr(edges), axis=1)

    top, bot, left, right = get_points(gray.shape, top_edge, bot_edge, left_edge, right_edge)
    c = ((top[0] + bot[0]) / 2, (left[1] + right[1]) / 2)
    r = (((right[0] - left[0]) / 2) + ((bot[1] - top[1]) / 2)) / 2
    logging.debug(f"Shape: {gray.shape[::-1]}")
    logging.debug(f"Top: {top}")
    logging.debug(f"Bottom: {bot}")
    logging.debug(f"Left: {left}")
    logging.debug(f"Right: {right}")
    logging.debug(f"Center: {c}")
    logging.debug(f"Radius: {r}")

    center, radius = get_circle(top, bot, left)

    return center, radius


def process_image(img):
    """Process image

    :param img: Image in BGR format
    :type img: np.ndarray
    """
    center, radius = get_center(img)
    icenter = tuple(map(lambda x: int(np.round(x)), center))

    output = img.copy()
    cv2.circle(output, icenter, int(np.round(radius)), (0, 0, 255), 2)
    cv2.line(output, (icenter[0], icenter[1] + 10), (icenter[0], icenter[1] - 10), (255, 128, 0), 1)
    cv2.line(output, (icenter[0] + 10, icenter[1]), (icenter[0] - 10, icenter[1]), (255, 128, 0), 1)
    cv2.imshow('circle', output)
    cv2.imwrite('Output/image.png', output)

    output = img.copy()
    output = output[150:250, 150:250]
    output = cv2.resize(output, (500, 500), interpolation=cv2.INTER_CUBIC)
    cv2.line(output,
             (5 * (icenter[0] - 150), 5 * (icenter[1] - 150) + 15),
             (5 * (icenter[0] - 150), 5 * (icenter[1] - 150) - 15),
             (255, 128, 0), 1)
    cv2.line(output,
             (5 * (icenter[0] - 150) + 15, 5 * (icenter[1] - 150)),
             (5 * (icenter[0] - 150) - 15, 5 * (icenter[1] - 150)),
             (255, 128, 0), 1)
    cv2.imshow('zoom', output)


image = cv2.imread('Input/image13.jpeg')
process_image(image)

# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# channels = cv2.split(hsv)
#
# cv2.imshow('hsv', np.vstack([channels[0], channels[1], channels[2]]))
#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# gray = cv2.bilateralFilter(gray, 5, 10, 2)
# cv2.imshow('filtered', gray)
#
# edges = cv2.Canny(gray, 0, 30)
# cv2.imshow('edges', edges)

# plt.show()
cv2.waitKey(0)
