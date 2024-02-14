def set_image(rgb, depth, coord, use_depth):
    height, width = coord[3]-coord[1], coord[2]-coord[0]
    minxy, maxxy = min(height, width), max(height, width)
    diff = maxxy - minxy

    x1, y1, x2, y2 = coord[0], coord[1], coord[2], coord[3]

    # Just keep the part of the image with the hand
    if maxxy == width: # if max == x
        rgb[y1-int(diff/2):y1, x1:x2] = 0
        depth[y1-int(diff/2):y1, x1:x2] = 0

        rgb[y2:y2+int(diff/2), x1:x2] = 0
        depth[y2:y2+int(diff/2), x1:x2] = 0

        rgb = rgb[y1-int(diff/2):y2+int(diff/2), x1:x2]
        depth = depth[y1-int(diff/2):y2+int(diff/2), x1:x2]
    
    else:
        rgb[y1:y2, x1-int(diff/2):x1] = 0
        depth[y1:y2, x1-int(diff/2):x1] = 0

        rgb[y1:y2, x2:x2+int(diff/2)] = 0
        depth[y1:y2, x2:x2+int(diff/2)] = 0

        rgb = rgb[y1:y2, x1-int(diff/2):x2+int(diff/2)]
        depth = depth[y1:y2, x1-int(diff/2):x2+int(diff/2)]
    
    if not use_depth:
        depth[:,:] = 0
    return rgb, depth