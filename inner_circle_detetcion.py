import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.signal import savgol_filter


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
    y_c = np.sqrt(r ** 2 - (x - center[0]) ** 2) + center[1]
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
    x = x[y != 0]
    y = y[y != 0]

    center = (100, 0)
    cutoff = np.linspace(50, 2.5, 10)  # Cutoff values are linear from high to low
    r = 0

    for i in range(10):
        diff, center, r = fit(x, y, center)
        # Cutoff Outliers
        x = x[abs(diff) < cutoff[i]]
        y = y[abs(diff) < cutoff[i]]

    y_c_orig = np.sqrt(r ** 2 - (np.arange(0, len(orig), dtype=np.uint16) - center[0]) ** 2) + center[1]

    # if args.show_plots:
    plt.figure()
    plt.plot(np.arange(0, len(orig)), orig)
    plt.plot(np.arange(0, len(orig)), y_c_orig)

    return np.array((np.nanargmax(y_c_orig), np.nanmax(y_c_orig)))


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

    # Calculate center of translated circle
    d = 2 * (b[0] * c[1] - b[1] * c[0])
    x0 = 1 / d * (c[1] * (b[0] ** 2 + b[1] ** 2) - b[1] * (c[0] ** 2 + c[1] ** 2))
    y0 = 1 / d * (b[0] * (c[0] ** 2 + c[1] ** 2) - c[0] * (b[0] ** 2 + b[1] ** 2))

    # Calculate radius
    radius = np.sqrt(x0 ** 2 + y0 ** 2)

    # Translate center back to actual location
    center = np.array([x0, y0])
    center += a
    return center, radius


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:
        center = (w // 2, h // 2)
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[0])

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


size = 100
CENTER = (size // 2, size // 2)
RADIUS = size // 4
blank = np.zeros((size, size), np.uint8)

cv2.circle(blank, CENTER, RADIUS, (255, 255, 255), 1)

rng = 255 * np.abs(np.round(np.random.normal(0, 0.2, (size, size))).astype(np.uint8))
cv2.imshow('noise', cv2.resize(rng, (500, 500), interpolation=cv2.INTER_NEAREST))
blank += rng

mask = create_circular_mask(size, size, CENTER, RADIUS)
mask_img = blank.copy()
mask_img[~mask] = 0
cv2.imshow('mask', cv2.resize(mask_img, (500, 500), interpolation=cv2.INTER_NEAREST))


left = blank[:, :size // 2]
right = blank[:, size // 2:]
top = blank[:size // 2, :]
bot = blank[size // 2:, :]
cv2.imshow('left', cv2.resize(left, (250, 500), interpolation=cv2.INTER_NEAREST))
cv2.imshow('right', cv2.resize(right, (250, 500), interpolation=cv2.INTER_NEAREST))
cv2.imshow('top', cv2.resize(top, (500, 250), interpolation=cv2.INTER_NEAREST))
cv2.imshow('bottom', cv2.resize(bot, (500, 250), interpolation=cv2.INTER_NEAREST))

left_edge = np.argmax(np.fliplr(left), axis=1)
right_edge = np.argmax(right, axis=1)
top_edge = np.argmax(np.flipud(top), axis=0)
bot_edge = np.argmax(bot, axis=0)

index = np.concatenate([np.r_[0:(size // 2 - size // 4)], np.r_[(size // 2 + size // 4):size]])
left_edge[index] = 0
right_edge[index] = 0
top_edge[index] = 0
bot_edge[index] = 0

left_savgol = savgol_filter(left_edge, 11, 2)
plt.figure()
plt.plot(np.arange(0, len(left_edge)), left_edge)
plt.plot(np.arange(0, len(left_edge)), left_savgol)

left_point = fit_circle(left_edge)[::-1]
left_point[0] = size // 2 - left_point[0]
right_point = np.array([size // 2, 0]) + fit_circle(right_edge)[::-1]
top_point = fit_circle(top_edge)
top_point[1] = size // 2 - top_point[1]
bot_point = np.array([0, size // 2]) + fit_circle(bot_edge)

print(left_point, right_point, top_point, bot_point)

center, radius = get_circle(top_point, left_point, bot_point)
print(center, radius)
print("Center diff:", center - np.array(CENTER))
print('Radius diff:', radius - RADIUS)

output = cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR)
cv2.circle(output, tuple(center.astype(int)), int(np.ceil(radius)), (0, 255, 0), 1)
cv2.circle(output, tuple(left_point.astype(int)), 1 + size // 125, (0, 0, 255), -1)
cv2.circle(output, tuple(right_point.astype(int)), 1 + size // 125, (0, 0, 255), -1)
cv2.circle(output, tuple(top_point.astype(int)), 1 + size // 125, (0, 0, 255), -1)
cv2.circle(output, tuple(bot_point.astype(int)), 1 + size // 125, (0, 0, 255), -1)
cv2.imshow('fit', cv2.resize(output, (500, 500), interpolation=cv2.INTER_NEAREST))

cv2.imshow('zoom', cv2.resize(blank, (500, 500), interpolation=cv2.INTER_NEAREST))

# plt.show()
cv2.waitKey(0)
