"""This python script gives the misalignment of the emitter and returns it in spherical coordinates

Author: Anshul Gupta

Usage:
As a script:
    python alignment.py [-h] [-c CONFIG] [-v] [s] [-sshow-extra] FILE

    positional arguments:
        FILE                        File to analyze

    optional arguments:
        -h, --help                  Show help message
        -c CONFIG, --config CONFIG  Provide config file
        -v, --verbose               Show debug log. This will show some inner workings
        -s, --show                  Show main openCV windows
        --show-extra                Show debug openCV windows. This will also enable -s flag

As a module:
    >>> import alignment as align
    >>> import cv2

    Config file must be provided before module can be used
    >>> align.process_config('config.ini')

    Use get_3d_rotation function to get 3d rotation
    >>> image = cv2.imread('image.png')
    >>> align.get_3d_rotation(image)

    if get_3d_rotation has out=True and save folder is defined, files will be stored in 'save_folder/import'

Config file (Options with values are recommended values):
    [FILE]
    save folder =                   Folder to save files to. Put as blank for no output
    save type = png                 Type to save images as
    put text =                      Whether to put misalignment value on image. yes/no

    [RIGHT OBJECT]
    x1 =                            Left x position
    y1 =                            Top y position
    x2 =                            Right x position
    y2 =                            Bottom y position

    [LEFT OBJECT]
    x1 =                            Left x position
    y1 =                            Top y position
    x2 =                            Right x position
    y2 =                            Bottom y position

    [CONTOUR DETECTION]             Options for contour detection (see openCV docs for better understanding)
    thresh: 90                      Threshold value for openCV::threshold
    maxVal: 255                     Maximum value to use with THRESH_BINARY threshold type for openCV::threshold
    type: 0                         Threshold type for openCV::threshold (see openCV docs)
    contour mode: 1                 Contour mode for openCV::findContours
    contour method: 2               Contour method for openCV::findContours
"""

import argparse
import configparser
import logging
import os

import cv2
import numpy as np

# Parse Arguments
parser = argparse.ArgumentParser(description="Checks if emitter is aligned vertically. Gives emitter alignment in "
                                             "spherical coordinates")
parser.add_argument('file', help='File to analyze')
parser.add_argument('-c', '--config', default='config.ini', help='Config file')
parser.add_argument('-v', '--verbose', action='store_true', help='Show debug log')
parser.add_argument('-s', '--show', action='store_true', help='Show main openCV windows')
parser.add_argument('--show-extra', action='store_true', help='Show debug openCV windows. -s will be enabled')

if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f'{args.config} could not be found')
else:
    args = parser.parse_args(["import", '-c', 'none'])

# If showing debug, show main ones too
if args.show_extra:
    args.show = True

np.set_printoptions(suppress=True)  # Prevent printing in scientific notation

if args.verbose:
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
else:
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# Setup config variables
save_folder, save_type, put_text = [None] * 3
right_angle, left_angle = [None] * 2
left_x1, left_y1, left_x2, left_y2, right_x1, right_y1, right_x2, right_y2 = [None] * 8
thresh_val, max_val, thresh_type, contour_mode, contour_method = [None] * 5

file_name = 'import'  # If file is imported


def process_config(config_file):
    """Initialize parameters using config file

    :param config_file: config file
    :type config_file: str
    """
    global save_folder, save_type, put_text
    global right_angle, left_angle
    global left_x1, left_y1, left_x2, left_y2, right_x1, right_y1, right_x2, right_y2
    global thresh_val, max_val, thresh_type, contour_mode, contour_method
    config = configparser.ConfigParser()
    config.read(config_file)

    save_folder = config.get('FILE', 'save folder')
    save_type = config.get('FILE', 'save type')
    put_text = config.getboolean('FILE', 'put text')

    right_angle = np.radians(35)
    left_angle = np.radians(-35)

    left_x1 = config.getint('LEFT OBJECT', 'x1')
    left_y1 = config.getint('LEFT OBJECT', 'y1')
    left_x2 = config.getint('LEFT OBJECT', 'x2')
    left_y2 = config.getint('LEFT OBJECT', 'y2')

    right_x1 = config.getint('RIGHT OBJECT', 'x1')
    right_y1 = config.getint('RIGHT OBJECT', 'y1')
    right_x2 = config.getint('RIGHT OBJECT', 'x2')
    right_y2 = config.getint('RIGHT OBJECT', 'y2')

    thresh_val = config.getint('CONTOUR DETECTION', 'thresh')
    max_val = config.getint('CONTOUR DETECTION', 'maxVal')
    thresh_type = config.getint('CONTOUR DETECTION', 'type')
    contour_mode = config.getint('CONTOUR DETECTION', 'contour mode')
    contour_method = config.getint('CONTOUR DETECTION', 'contour method')


def sort_contours(cnts, method='left-to-right'):
    """Sort contours in certain direction

    :param cnts: contours
    :type cnts: np.ndarray
    :param method: sorting order. 'left-to-right' or 'right-to-left' or 'top-to-bottom' or 'bottom-to-top'
    :type method: str
    :return: sorted contours
    :rtype: np.ndarray
    """
    reverse = False
    i = 0
    if method == 'right-to-left' or method == 'bottom-to-top':
        reverse = True
    if method == 'top-to-bottom' or method == 'bottom-to-top':
        i = 1
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, bounding_boxes) = zip(*sorted(zip(cnts, bounding_boxes), key=lambda b: b[1][i], reverse=reverse))

    return cnts


def alignment(img, y1, y2, x1, x2, out=None):
    """Get misalignment of emitter

    :param img: Image in BGR format
    :type img: np.ndarray
    :param y1: top y position
    :type y1: int
    :param y2: bottom y position
    :type y2: int
    :param x1: left x position
    :type x1: int
    :param x2: right x position
    :type x2: int
    :param out: numpy array to write output to. leave as None for no output
    :type out: np.ndarray
    :return: degree of misalignment
    :rtype: float
    :raises AssertionError: if filling leaves more or less than 1 contour
    """
    img = img[y1:y2, x1:x2]  # Crop image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    # gray = cv2.equalizeHist(gray)  # Histogram Equalization

    ret, thresh = cv2.threshold(gray, thresh_val, max_val, thresh_type)  # Threshold image
    contours, _ = cv2.findContours(thresh, contour_mode, contour_method)  # Get contours of image
    contours = sort_contours(contours)  # Sort contours from left to write

    if args.show_extra:
        cv2.imshow('Original Thresh', thresh)

    for i in range(1, len(contours) - 1):  # Fill in all contours except edge ones
        cnt = contours[i]
        cv2.fillPoly(thresh, [cnt], color=(0, 0, 0))

    thresh = cv2.bitwise_not(thresh)  # Flip image so emitter becomes white and background black
    contours, _ = cv2.findContours(thresh, 1, 2)  # Find contours of new filled in image

    assert len(contours) == 1  # This means the filling went wrong

    if args.show_extra:
        cv2.imshow('Thresh', thresh)

    # Fit rectangle around emitter
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(np.around(box))
    logging.debug(f'Bounding Box: {box}')

    degree = np.degrees(np.arctan2(box[0][0] - box[1][0], box[0][1] - box[1][1]))

    if out is not None:
        overlay = img.copy()
        cv2.drawContours(overlay, [box], 0, (0, 255, 0), -1)
        out[...] = cv2.addWeighted(overlay, 0.3, img, 1 - 0.3, 0)
        if put_text:
            cv2.putText(out, str(np.around(degree, 2)), (1, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
                        cv2.LINE_AA)

    return degree


def get_3d_rotation(img, out=False):
    """Get 3d rotation of emitter in image

    :param img: full image in BGR format
    :type img: np.ndarray
    :param out: to have output or not
    :type out: bool
    :return: (horizontal, vertical)
    :rtype: (float, float)
    """
    if out or args.show:
        # Setup output images
        output_left = np.zeros((left_y2 - left_y1, left_x2 - left_x1, 3), np.uint8)
        output_right = np.zeros((right_y2 - right_y1, right_x2 - right_x1, 3), np.uint8)
    else:
        output_left = None
        output_right = None

    left = alignment(img, left_y1, left_y2, left_x1, left_x2, output_left)
    logging.info(f'Left misalignment:  {left}°')

    right = alignment(img, right_y1, right_y2, right_x1, right_x2, output_right)
    logging.info(f'Right misalignment: {right}°')

    if out:
        # Save images
        left_path = os.path.join(save_folder, file_name, 'left_' + str(np.around(left, 2)) + '.' + save_type)
        right_path = os.path.join(save_folder, file_name, 'right_' + str(np.around(right, 2)) + '.' + save_type)
        if not cv2.imwrite(left_path, output_left):
            logging.error(f'Could not write to {left_path}')
        if not cv2.imwrite(right_path, output_right):
            logging.error(f'Could not write to {right_path}')

    if args.show:
        cv2.imshow('Left', output_left)
        cv2.imshow('Right', output_right)

    horizontal = right * np.cos(right_angle) + left * np.cos(left_angle)
    vertical = right * np.sin(right_angle) + left * np.sin(left_angle)
    return horizontal, vertical


if args.file != 'import':
    try:
        process_config(args.config)
    except configparser.NoSectionError:
        raise

    image = cv2.imread(args.file)
    if image is None:
        raise FileNotFoundError(f'{args.file} not found')

    if save_folder:
        file_name = os.path.split(args.file)[-1]
        try:
            os.mkdir(os.path.join(save_folder, file_name))
        except FileExistsError:
            pass
        hor, ver = get_3d_rotation(image, True)
    else:
        hor, ver = get_3d_rotation(image)

    logging.info(f'Horizontal 3D Rotation: {hor}°')
    logging.info(f'Vertical 3D Rotation:   {ver}°')

    if args.show:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
