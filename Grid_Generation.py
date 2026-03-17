import numpy as np

# global cache dictionary

def get_grid(shape):

    grid_cache = {}
    # if grid already computed for this shape, reuse it
    if shape in grid_cache:
        return grid_cache[shape]

    h, w = shape  # wafer height and width

    # create coordinate grid
    yy, xx = np.mgrid[0:h, 0:w]

    # compute wafer center
    cy = (h - 1) / 2
    cx = (w - 1) / 2

    # normalize coordinates
    y_norm = (yy - cy) / max(cy, 1e-9)
    x_norm = (xx - cx) / max(cx, 1e-9)

    # radial distance from center
    r = np.sqrt(x_norm**2 + y_norm**2)

    # store result in cache
    grid_cache[shape] = (x_norm, y_norm, r)

    return grid_cache[shape]
