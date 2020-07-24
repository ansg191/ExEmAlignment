"""This python script detects the outer most circle and gives its center to compare it to the emitter.

Author: Anshul Gupta

Usage:
As a script:
    python concentricity.py [-h] [-c CONFIG] [-s] [--show-extra] [-v] [--show-ignored] [-p] FILE

    positional arguments:
        FILE                        File to analyze

    optional arguments:
        -h, --help                  Show help message
        -c CONFIG, --config CONFIG  Provide config file
        -s, --show                  Show main openCV windows
        --show-extra                Show debug openCV windows. This will also enable -s flag
        -v, --verbose               Show debug log. This will show some of the inner workings
        --show-ignored              Show ignored warnings. Enabling this will show the suppressed runtime warnings
        -p, --show-plots            Show plots of curve fitting. Use this to debug poor curve fits

As a module:
    >>> import concentricity as con
    >>> import cv2

    Config file must be provided before module can be used
    >>> con.process_config('config.ini')

    Use process_image function to get center and radius
    >>> image = cv2.imread('image.png')
    >>> con.process_image(image)

    If save folder is defined in config file, files will be stored in 'save_folder/import'

Config file (Options with values are recommended values):
    [CROP]                          Options for cropping image
    x1 =                            Left x position, leave x1 blank for no crop and to ignore rest of CROP options
    x2 =                            Right x position
    y1 =                            Top y position
    y2 =                            Bottom y position

    [EDGE DETECTION]                Options for Edge Detection
    filter = 30                     Diameter of pixel neighborhood for bilateral filter, std of color space is filter*2,
                                    std of coord space is filter / 2
    edge low = 150                  Canny Edge Detection minVal
    edge high = 200                 Canny Edge Detection maxVal

    [CIRCLE FIT]                    Options for Fitting the Circle
    high cutoff = 50                Starting Cutoff residual. Lower values will aggressively cutoff more values.
                                    Higher values may require more runs to remove outliers
    low cutoff = 2.5                Final Cutoff residual. Lower values may cutoff good values. Higher values may
                                    leave outliers in final fit
    runs = 10                       Number of residual cuts and refits. More runs increase processing time
    method = 3                      Method for determining final circle. 3 uses a circumcircle, which produces better
                                    results. 4 uses cardinal points which may be more inaccurate

    [OUTPUT]                        Options for Output images
    zoom size = 100                 Size to zoom into. Zoom will be a square in the center of the picture
    save folder =                   Folder to save files to. Put as blank for no output
    save type = png                 Type to save images as
"""

import argparse
import configparser
import logging
import os
import warnings

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize

# Parse Arguments
parser = argparse.ArgumentParser(description="Checks if emitter is located at the center of the outer circle")
parser.add_argument('file', help='File to analyze')
parser.add_argument('-c', '--config', default='concentricity.ini', help='Config file')
parser.add_argument('-s', '--show', action='store_true', help='Show main openCV windows')
parser.add_argument('--show-extra', action='store_true', help='Show debug openCV windows. -s will be enabled')
parser.add_argument('-v', '--verbose', action='store_true', help='Show debug log')
parser.add_argument('--show-ignored', action='store_true', help='Show ignored warnings')
parser.add_argument('-p', '--show-plots', action='store_true', help='Show plots of curve fitting')

if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f'{args.config} could not be found')
else:
    args = parser.parse_args(["import", '-c', 'none'])

# if showing debug, the regular ones should be shown too
if args.show_extra:
    args.show = True

if not args.show_ignored:
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # Filter Runtime warnings

np.set_printoptions(suppress=True)  # Prevent printing in scientific notation

if args.verbose:
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
else:
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# Setup config variables
x1, x2, y1, y2, filter_val, edge_low, edge_high, \
high_cutoff, low_cutoff, runs, method, zoom_size, save_folder, save_type = [None] * 14

file_name = 'import'


def process_config(file):
    """Initialize parameters using config file

    :param file: config file
    :type file: str
    """
    global x1, x2, y1, y2, filter_val, edge_low, edge_high, high_cutoff, low_cutoff, runs, method, zoom_size, \
        save_folder, save_type
    # Read config file
    config = configparser.ConfigParser()
    config.read(file)

    try:
        x1 = config.getint('CROP', 'x1')
    except ValueError:
        x1 = None
    else:
        x2 = config.getint('CROP', 'x2')
        y1 = config.getint('CROP', 'y1')
        y2 = config.getint('CROP', 'y2')

    filter_val = config.getint('EDGE DETECTION', 'filter')
    edge_low = config.getint('EDGE DETECTION', 'edge low')
    edge_high = config.getint('EDGE DETECTION', 'edge high')

    high_cutoff = config.getfloat('CIRCLE FIT', 'high cutoff')
    low_cutoff = config.getfloat('CIRCLE FIT', 'low cutoff')
    runs = config.getint('CIRCLE FIT', 'runs')
    method = config.getint('CIRCLE FIT', 'method')

    zoom_size = config.getint('OUTPUT', 'zoom size')
    save_folder = config.get('OUTPUT', 'save folder')
    save_type = config.get('OUTPUT', 'save type')

    if save_folder:
        try:
            os.mkdir(os.path.join(save_folder, 'import'))  # Create output folder
        except FileExistsError:
            pass


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
    :rtype: (np.ndarray, (float, float), float)
    """
    # noinspection PyTupleAssignmentBalance,PyTypeChecker
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
    orig = y.copy()

    x = np.arange(0, len(y), dtype=np.uint16)

    center = (200, 200)
    cutoff = np.linspace(high_cutoff, low_cutoff, runs)  # Cutoff values are linear from high to low
    r = 0

    for i in range(runs):
        diff, center, r = fit(x, y, center)
        # Cutoff Outliers
        x = x[abs(diff) < cutoff[i]]
        y = y[abs(diff) < cutoff[i]]

    y_c_orig = -np.sqrt(r ** 2 - (np.arange(0, len(orig), dtype=np.uint16) - center[0]) ** 2) + center[1]

    if args.show_plots:
        plt.figure()
        plt.plot(np.arange(0, len(orig)), orig)
        plt.plot(np.arange(0, len(orig)), y_c_orig)

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
    logging.info(f"Radius: {radius}\n")
    return center, radius


def get_center(img):
    """Gets the center and radius of outer circle in an image

    :param img: Image in BGR format
    :type img: np.ndarray
    :return: (center, radius)
    :rtype: (np.ndarray, float)
    """
    gray: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    gray = cv2.equalizeHist(gray)  # Equalize histogram and contrast
    gray = cv2.bilateralFilter(gray, filter_val, filter_val * 2, filter_val // 2)  # Filter to remove noise
    edges: np.ndarray = cv2.Canny(gray, edge_low, edge_high)  # Edge Detection

    if args.show_extra:
        # Overlay edges on image
        output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        output = cv2.add(img, output)
        cv2.imshow('output', output)
        cv2.imshow('equal', gray)
        cv2.imshow('edges', edges)

    # Scan in each direction for closest edge
    top_edge = np.argmax(edges, axis=0)
    bot_edge = np.argmax(np.flipud(edges), axis=0)
    left_edge = np.argmax(edges, axis=1)
    right_edge = np.argmax(np.fliplr(edges), axis=1)

    # Get cardinal points of circle
    top, bot, left, right = get_points(gray.shape, top_edge, bot_edge, left_edge, right_edge)

    logging.debug(f"Shape: {gray.shape[::-1]}")
    logging.debug(f"Top: {top}")
    logging.debug(f"Bottom: {bot}")
    logging.debug(f"Left: {left}")
    logging.debug(f"Right: {right}\n")

    if method == 4:
        # Method that uses 4 points. Less accurate because it assumes points are at perfect 90 degree intervals.
        center = np.array((top[0] + bot[0]) / 2, (left[1] + right[1]) / 2)
        radius = (((right[0] - left[0]) / 2) + ((bot[1] - top[1]) / 2)) / 2
        logging.debug(f"Center and Radius using 4 points")
        logging.info(f"Center: {center}")
        logging.info(f"Radius: {radius}\n")
    elif method == 3:
        # Method that uses 3 points. Better as it produces the circumcircle of the triangle.
        logging.debug(f"Center and Radius using 3 points")
        center, radius = get_circle(top, bot, left)
    else:
        raise ValueError(f'Invalid Method {method}')

    return center, radius


def process_image(img):
    """Process image

    :param img: Image in BGR format
    :type img: np.ndarray
    """
    if method is None:
        raise RuntimeError("Config has not been initialized. Initialize using process_config")

    if x1:
        img = img[y1:y2, x1:x2]  # Crop image
        if args.show_extra:
            cv2.imshow('Crop', img)

    center, radius = get_center(img)  # Get center and radius
    icenter = tuple(np.round(center).astype(int))  # Convert to integer for openCV

    # Put circle on image
    output = img.copy()
    cv2.circle(output, icenter, int(np.round(radius)), (0, 0, 255), 2)
    cv2.line(output, (icenter[0], icenter[1] + 10), (icenter[0], icenter[1] - 10), (255, 128, 0), 1)
    cv2.line(output, (icenter[0] + 10, icenter[1]), (icenter[0] - 10, icenter[1]), (255, 128, 0), 1)
    if args.show:
        # Show image
        cv2.imshow('circle', output)
    if save_folder:
        # Save image
        save_file = os.path.join(save_folder, file_name, f'circle.{save_type}')
        if not cv2.imwrite(save_file, output):
            logging.error(f'Could not write to {save_file}')
        else:
            logging.debug(f'Wrote circle image to {save_file}')

    # Zoom into center
    # Get crop coordinates
    x_min = (img.shape[1] // 2) - (zoom_size // 2)
    y_min = (img.shape[0] // 2) - (zoom_size // 2)
    scale = 500 // zoom_size  # Get scaling factor

    output = img.copy()
    output = output[y_min:y_min + zoom_size, x_min:x_min + zoom_size]  # Crop
    output = cv2.resize(output, (500, 500), interpolation=cv2.INTER_CUBIC)  # Resize to make larger

    # Draw plus sign at center
    cv2.line(output,
             (scale * (icenter[0] - x_min), scale * (icenter[1] - y_min) + 15),
             (scale * (icenter[0] - x_min), scale * (icenter[1] - y_min) - 15),
             (255, 128, 0), 1)
    cv2.line(output,
             (scale * (icenter[0] - x_min) + 15, scale * (icenter[1] - y_min)),
             (scale * (icenter[0] - x_min) - 15, scale * (icenter[1] - y_min)),
             (255, 128, 0), 1)

    if args.show:
        # Show image
        cv2.imshow('zoom', output)
    if save_folder:
        # Save image
        save_file = os.path.join(save_folder, file_name, f'zoom.{save_type}')
        if not cv2.imwrite(save_file, output):
            logging.error(f'Could not write to {save_file}')
        else:
            logging.debug(f'Wrote zoom image to {save_file}')


if args.file != 'import':
    try:
        process_config(args.config)
    except configparser.NoSectionError:
        logging.error('Config file failed to parse')
    image = cv2.imread(args.file)  # Read image
    if args.show_extra:
        cv2.imshow('Image', image)

    if save_folder:
        file_name = os.path.split(args.file)[-1]
        try:
            os.mkdir(os.path.join(save_folder, file_name))  # Create output folder
        except FileExistsError:
            pass

    process_image(image)  # Process image

    if args.show_plots:
        plt.show()
    if args.show:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
