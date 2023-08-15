import numpy as np
import cv2

def set_target(index):
    """
    Return an array of 0 and 1, the position of 1 in the array indicates the class of the corresponding image

    :param index: int the index of the class
    :return: np.ndarray of 0 and 1
    """
    output = np.zeros(29)
    output[index] = 1

def set_path_and_coord(rgb_image, depth_image, coord):
    """
    Return an array containing every infomation needed to find and cut correctly an image

    :param rgb_image: string path to the rgb image
    :param depth_image: string path to the depth image
    :param coord: coord of the image where the hand is
    :return: the array with every information
    """

    output = np.zeros(6, dtype=object)
    output[0] = rgb_image
    output[1] = depth_image
    output[2:] = coord[1:len(coord)-1].split()
    
    return output

def set_image(rgb, depth, x1, x2, y1, y2, width, height, use_depth):
    minxy, maxxy = min(height, width), max(height, width)
    diff = maxxy - minxy

    # Just keep the part of the image with the hand
    if maxxy == width: # if max == x
        rgb[x1:x2, y1-int(diff/2):y1] = 0
        depth[x1:x2, y1-int(diff/2):y1] = 0

        rgb[x1:x2, y2:y2+int(diff/2)] = 0
        depth[x1:x2, y2:y2+int(diff/2)] = 0

        rgb = rgb[x1:x2, y1-int(diff/2):y2+int(diff/2)]
        depth = depth[x1:x2, y1-int(diff/2):y2+int(diff/2)]
    
    else:
        rgb[x1-int(diff/2):x1, y1:y2] = 0
        depth[x1-int(diff/2):x1, y1:y2] = 0

        rgb[x2:x2+int(diff/2), y1:y2] = 0
        depth[x2:x2+int(diff/2), y1:y2] = 0

        rgb = rgb[x1-int(diff/2):x2+int(diff/2), y1:y2]
        depth = depth[x1-int(diff/2):x2+int(diff/2), y1:y2]
    
    if not use_depth:
        depth[:,:] = 0
    return rgb, depth

def reduce_resolution(image, resol):
    # Récupérer les dimensions de l'image
    height, width = image.shape[:2]

    # Calculer les nouvelles dimensions de l'image réduite
    new_width = int(width * resol)
    new_height = int(height * resol)

    # Réduire la résolution de l'image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_image
     