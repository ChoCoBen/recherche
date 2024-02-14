import numpy as np

def set_coord(coord):
    """
    coord is a numpy array with y1, x1, height, width. We want x1, y1, x2, y2
    """
    new_coord = np.zeros(4)
    new_coord[0] = coord[0]
    new_coord[1] = coord[1]

    height = coord[3]
    width = coord[2]

    new_coord[2] = coord[0] + width
    new_coord[3] = coord[1] + height

    return new_coord
