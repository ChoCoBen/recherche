import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

def calculate_mediapipe_BB(results, w, h, coef):
    """
    Return a list of bounding boxes, those bounding boxes are a list like [x1, y1, x2, y2] with (x1<x2, y1<y2)
    """
    bounding_boxes = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            maxlmx, maxlmy = 0, 0
            minlmx, minlmy = 1, 1

            # For every key points for each hand
            for id, lm in enumerate(hand_landmarks.landmark):
                maxlmx = lm.x if lm.x > maxlmx else maxlmx
                minlmx = lm.x if lm.x < minlmx else minlmx
                maxlmy = lm.y if lm.y > maxlmy else maxlmy
                minlmy = lm.y if lm.y < minlmy else minlmy
            
            first_bounding_box = [int(minlmx * w),
                                    int(minlmy * h),
                                    int(maxlmx * w),
                                    int(maxlmy * h)]
            
            w_r = first_bounding_box[2] - first_bounding_box[0]
            h_r = first_bounding_box[3] - first_bounding_box[1]

            sd_bounding_box = [first_bounding_box[0] - int(coef * w_r),
                                first_bounding_box[1] - int(coef * h_r),
                                first_bounding_box[2] + int(coef * w_r),
                                first_bounding_box[3] + int(coef * h_r)] # x1, y1, x2, y2 (x1<x2, y1<y2)
            
            bounding_boxes.append(sd_bounding_box)
    return bounding_boxes

def cut_image_BB(image, BB):
    bounding_box = BB
    y1, x1, y2, x2 = BB[0], BB[1], BB[2], BB[3]
    height, width = y2 - y1, x2 - x1

    minxy, maxxy = min(height, width), max(height, width)
    diff = maxxy - minxy

    # Just keep the part of the image with the hand
    if maxxy == width: # if width > height
        image[x1:x2, y1-int(diff/2):y1] = 0

        image[x1:x2, y2:y2+int(diff/2)] = 0

        image = image[x1:x2, y1-int(diff/2):y2+int(diff/2)]
    
    else:
        image[x1-int(diff/2):x1, y1:y2] = 0

        image[x2:x2+int(diff/2), y1:y2] = 0

        image = image[x1-int(diff/2):x2+int(diff/2), y1:y2]

    return image

def setup_image_for_model(image, BB):
    """
    Return a tensor of an image of size (100, 100) with 4 channels, one for depth (equal to 0) because the model needs 4 channels
    Color image is swap from RGB to BGR
    The image is normalized as it is for the model
    """
    image = cut_image_BB(image, BB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # model trained with BGR images

    image = np.asarray(cv2.resize(image, (100,100)))
    depth = np.zeros((100,100))
    image = image = np.concatenate((image, depth[..., None]), axis=2)
    image = np.transpose(image, (2, 0, 1))

    image[0:3,:] /= 255.0
    image[3,:] /= 65535.0
    image /= image.max()

    image = torch.tensor(image).to(torch.float)
    return image

def display_image_BB(image):
    """
    Take an image rgbd tensor type, shape: 4,x,x and normalized.
    Display this image
    """
    images_rgb = image[0:3,:]
    images_rgb *= 255
    images_rgb = images_rgb.numpy().astype(np.uint8)

    images_rgb = np.transpose(images_rgb, (1,2,0))

    fig, ax = plt.subplots(1)
    ax.imshow(images_rgb)

    plt.show()

def display_rect_BB(image, BB):
    pass