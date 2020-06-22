import configparser

import cv2
import numpy as np
import glob

config = configparser.ConfigParser()
config.read('config.ini')

file = config.get('FILE', 'file')
folder = config.get('FILE', 'folder')
file_type = config.get('FILE', 'file type')

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
normal_contour_number = config.getint('CONTOUR DETECTION', 'normal contour number')
nth_contour = config.getint('CONTOUR DETECTION', 'nth contour')


def check_alignment(img, y1, y2, x1, x2):
    img = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, thresh_val, max_val, thresh_type)
    contours, hierarchy = cv2.findContours(thresh, contour_mode, contour_method)
    contours = sort_contours(contours)

    # if len(contours) != normal_contour_number:
    print(len(contours))
    cv2.drawContours(img, [contours[nth_contour - 1]], 0, (0, 0, 255), 2)
    cv2.drawContours(img, [contours[nth_contour]], 0, (0, 255, 0), 2)
    cv2.drawContours(img, [contours[nth_contour + 1]], 0, (255, 0, 0), 2)
    cv2.imshow('debug', img)
    cv2.waitKey(0)
    # raise ValueError('Abnormal number of contours found. Found ' + str(len(contours)))

    cnt = contours[nth_contour]

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    return np.degrees(np.arctan2(box[0][0] - box[1][0], box[0][1] - box[1][1]))


def sort_contours(cnts, method='left-to-right'):
    reverse = False
    i = 0
    if method == 'right-to-left' or method == 'bottom-to-top':
        reverse = True
    if method == 'top-to-bottom' or method == 'bottom-to-top':
        i = 1
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, bounding_boxes) = zip(*sorted(zip(cnts, bounding_boxes), key=lambda b: b[1][i], reverse=reverse))

    return cnts


if not folder:
    image = cv2.imread(file)
    left = check_alignment(image, left_y1, left_y2, left_x1, left_x2)
    right = check_alignment(image, right_y1, right_y2, right_x1, right_x2)
    print("Left emitter alignment: ", left, "degrees")
    print("Right emitter alignment:", right, "degrees")
else:
    files = glob.glob(folder + '/*.' + file_type)
    for file in files:
        print("Checking alignment on", file)
        image = cv2.imread(file)
        left = check_alignment(image, left_y1, left_y2, left_x1, left_x2)
        right = check_alignment(image, right_y1, right_y2, right_x1, right_x2)
        print("\tLeft emitter alignment: ", left, "degrees")
        print("\tRight emitter alignment:", right, "degrees")
