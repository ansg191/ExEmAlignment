import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
import configparser

from matplotlib import pyplot as plt

config = configparser.ConfigParser()



filename = "image.jpg"
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def find_angle(gray, y0, y1, x0, x1, min_val=30, max_val=100, outlier_threshold=5):
    gray = gray[y0:y1, x0:x1]  # Crop

    # image = cv2.GaussianBlur(gray, (0, 0), 3)
    # gray = cv2.

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, min_val, max_val)  # Get edges

    cv2.imshow('blurred', gray)
    cv2.imshow('edges', edges)
    cv2.resizeWindow('blurred', 800, 800)
    cv2.resizeWindow('edges', 800, 800)
    cv2.waitKey(0)

    edge = np.argmax(edges, axis=1)  # Get left edge

    # with warnings.catch_warnings():
    #     # Numpy gives warning if line has no variation. This just catches the warning.
    #     warnings.simplefilter('ignore')
    #     z = np.abs(np.nan_to_num(stats.zscore(edge)))  # Get z-score to remove outliers
    #
    # edge: np.ndarray = edge[z < outlier_threshold]  # Remove outliers

    x = np.arange(0, len(edge), dtype=np.uint16)  # y-pixel values for regression

    plt.plot(x, edge, 'o')

    model = LinearRegression()
    model.fit(x.reshape((-1, 1)), edge.reshape((-1, 1)))
    plt.plot(x, model.predict(x.reshape((-1, 1))).flatten())

    return np.degrees(np.arctan(model.coef_))


print(find_angle(gray, 255, 350, 725, 800))
print(find_angle(gray, 240, 335, 50, 125))
print(find_angle(gray, 70, 250, 680, 830))

plt.show()
