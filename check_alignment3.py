import cv2
import numpy as np
import configparser
import glob
import os

np.set_printoptions(suppress=True)

config = configparser.ConfigParser()
config.read('config.ini')

file = config.get('FILE', 'file')
folder = config.get('FILE', 'folder')
file_type = config.get('FILE', 'file type')
save_folder = config.get('FILE', 'save folder')
save_type = config.get('FILE', 'save type')
put_text = config.getboolean('FILE', 'put text')

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


def alignment(img, y1, y2, x1, x2, out=None):
    img = img[y1:y2, x1:x2]  # Crop image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    # gray = cv2.equalizeHist(gray)  # Histogram Equalization

    ret, thresh = cv2.threshold(gray, thresh_val, max_val, thresh_type)  # Threshold image
    contours, _ = cv2.findContours(thresh, contour_mode, contour_method)  # Get contours of image
    contours = sort_contours(contours)  # Sort contours from left to write

    for i in range(1, len(contours) - 1):  # Fill in all contours except edge ones
        cnt = contours[i]
        cv2.fillPoly(thresh, [cnt], color=(0, 0, 0))

    thresh = cv2.bitwise_not(thresh)  # Flip image so emitter becomes white and background black
    contours, _ = cv2.findContours(thresh, 1, 2)  # Find contours of new filled in image

    assert len(contours) == 1  # This means the filling went wrong

    # Fit rectangle around emitter
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(np.around(box))
    # print(box)

    degree = np.degrees(np.arctan2(box[0][0] - box[1][0], box[0][1] - box[1][1]))

    if out is not None:
        overlay = img.copy()
        cv2.drawContours(overlay, [box], 0, (0, 255, 0), -1)
        out[...] = cv2.addWeighted(overlay, 0.3, img, 1 - 0.3, 0)
        if put_text:
            cv2.putText(out, str(np.around(degree, 2)), (1, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
                        cv2.LINE_AA)

    return degree


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


if folder:
    files = glob.glob(folder + '/*' + file_type)
    for f in files:
        print("Checking alignment on", f)
        file_name = os.path.splitext(os.path.split(f)[1])[0]
        image = cv2.imread(f)
        output_left = np.zeros((left_y2 - left_y1, left_x2 - left_x1, 3), np.uint8)
        output_right = np.zeros((right_y2 - right_y1, right_x2 - right_x1, 3), np.uint8)
        left = alignment(image, left_y1, left_y2, left_x1, left_x2, output_left)
        right = alignment(image, right_y1, right_y2, right_x1, right_x2, output_right)
        if save_folder:
            try:
                os.mkdir(os.path.join(save_folder, file_name))
            except FileExistsError:
                pass

            if not cv2.imwrite(os.path.join(save_folder, file_name, 'left_' + str(np.around(left, 2)) + '.' +
                                                                    file_type),
                               output_left):
                raise IOError("Could not save output")
            if not cv2.imwrite(os.path.join(save_folder, file_name, 'right_' + str(np.around(right, 2)) + '.' +
                                                                    file_type),
                               output_right):
                raise IOError('Could not save output')
        print("\tLeft emitter alignment: ", left, "degrees")
        print("\tRight emitter alignment:", right, "degrees")
else:
    image = cv2.imread(file)
    file_name = os.path.splitext(os.path.split(file)[1])[0]
    output_left = np.zeros((left_y2 - left_y1, left_x2 - left_x1, 3), np.uint8)
    output_right = np.zeros((right_y2 - right_y1, right_x2 - right_x1, 3), np.uint8)
    left = alignment(image, left_y1, left_y2, left_x1, left_x2, output_left)
    right = alignment(image, right_y1, right_y2, right_x1, right_x2, output_right)
    if save_folder:
        cv2.imwrite(save_folder + '/' + file_name + '/left.' + file_type, output_left)
        cv2.imwrite(save_folder + '/' + file_name + '/right.' + file_type, output_right)
    print("Left emitter alignment: ", left, "degrees")
    print("Right emitter alignment:", right, "degrees")
