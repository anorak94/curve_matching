
from scipy.ndimage import gaussian_filter, zoom, map_coordinates
import numpy as np
import matplotlib.pyplot as plt
import math

def plotx(im1, im2):
    f, a = plt.subplots(1, 2, figsize  = (8, 8))
    a[0].imshow(im1)
    a[0].set_axis_off()
    a[1].imshow(im2)
    a[1].set_axis_off()
    plt.show()
    
    

## utilities function 
def change_resolution(img, resolution, sigma, order=1, ndim = 2):
    """
    change image's resolution
    Parameters
    ----------
    resolution : int
        how much to magnify
        if resolution is 2, the shape of the image will be halved
    sigma : float
        standard deviation of gaussian filter for smoothing
    order : int
        order of interpolation
    Returns
    -------
    img : ScalarImage
        zoomed scalar image
    """
    if resolution != 1:
        blurred_data = gaussian_filter(img, sigma)
        ratio = [1 / float(resolution)] * ndim
        data = zoom(blurred_data, ratio, order=order)
    elif resolution == 1:
        data = gaussian_filter(img, sigma)
    return data

def change_scale(img, maximum_value):
    data = maximum_value * img / np.max(img)
    return data

def warp_image(warp_a , image, order = 3):
    return map_coordinates(warp_a, image, order = 3)

def zoom_grid(grid, resolution, ndim,shape0  ):
    shape = grid.shape[1:]
    if resolution != 1:
        interpolated_grid = np.zeros((ndim,) + shape0)
        for i in range(ndim):
            interpolated_grid[i] = interpolate_mapping(
                grid[i], np.array(shape0, dtype=np.int32)
            ) * (shape0[i] - 1) / (shape[i] - 1)
        return interpolated_grid
    else:
        return grid

def interpolate_mapping(func,  target_shape):
    return interpolate2d(func, func.shape[0], func.shape[1], target_shape)
    
def identity_mapping(shape):
    x1 = [float(i) for i in range(shape[0])]
    y1 = [float(i)for i in range(shape[1])]
    return np.meshgrid(x1, y1, indexing = "xy")
    



def interpolate2d( func,  xlen_now,  ylen_now,  target_shape):
    xlen_target = target_shape[0]
    ylen_target = target_shape[1]

    interpolated = np.zeros((xlen_target, ylen_target))

    for x in range(xlen_target):
        xi = x * (xlen_now - 1) / (xlen_target - 1.)
        for y in range(ylen_target):
            yi = y * (ylen_now - 1) / (ylen_target - 1.)
            interpolated[x,y] = bilinear_interpolation(func, xi, yi, xlen_now, ylen_now)

    return interpolated


def bilinear_interpolation(func, x,  y,  xlen,  ylen):
    """
    Bilinear interpolation at a given position in the image.
    Parameters
    ----------
    func : double array
        Input function.
    x, y : double
        Position at which to interpolate.
    Returns
    -------
    value : double
        Interpolated value.
    """

    
    x0 = math.floor(x)
    x1 = math.ceil(x)
    y0 = math.floor(y)
    y1 = math.ceil(y)

    dx = x - x0
    dy = y - y0

    f0 = (1 - dy) * getValue2d(func, x0, y0, xlen, ylen, 'N') + dy * getValue2d(func, x0, y1, xlen, ylen, 'N')
    f1 = (1 - dy) * getValue2d(func, x1, y0, xlen, ylen, 'N') + dy * getValue2d(func, x1, y1, xlen, ylen, 'N')

    return (1 - dx) * f0 + dx * f1


def getValue2d(func,  x,  y,  xlen,  ylen,  mode='N'):
    if mode == 'N':
        if x < 0:
            x = 0
        elif x > xlen - 1:
            x = xlen - 1

        if y < 0:
            y = 0
        elif y > ylen - 1:
            y = ylen - 1
    elif mode == 'C':
        if x < 0 or x > xlen - 1 or y < 0 or y > ylen - 1:
            return 0
    
    return func[x , y]

def show_warp_field(grid, interval=1, shape = (120, 120), size = (8, 8),limit_axis=True, show_axis=False, plot_separately = True):
    
    if plot_separately:
        f, a = plt.subplots(1, 2, figsize = size)
        
        for x in range(0, shape[0], interval):
            a[0].set_axis_off()
            a[0].plot(grid[1, x, :], grid[0, x, :], 'k')
            a[0].invert_yaxis()
            a[0].set_title("x field")
            a[0].set_aspect("equal")
        for y in range(0, shape[1], interval):
            a[1].set_axis_off()
            a[1].set_title("y field")
            a[1].plot(grid[1, :, y], grid[0, :, y], 'r')
            a[1].invert_yaxis()
            a[1].set_aspect("equal")
        plt.show()
    else:
        plt.figure(figsize = size)
        if show_axis is False:
            plt.axis('off')
        ax = plt.gca()
        ax.invert_yaxis()
        ax.set_aspect('equal')
        for x in range(0, shape[0], interval):
            try:
                plt.plot(grid[1, x, :], grid[0, x, :], 'k')
            except:
                pass
        for y in range(0, shape[1], interval):
            try:
                plt.plot(grid[1, :, y], grid[0, :, y], 'k')
            except:
                pass
        plt.show()
