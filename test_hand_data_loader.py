from src import hand_gesture_dataset
from src.utils.data_augmentation import gaussian_noise, random_blocks, rotation
from os.path import join
import cv2
import numpy as np
from src.utils.dataset_helper import *
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Setting some values tu use the function as in the dataloader
cfg = {
    'dataset_path': '/home/travail/Dataset/ndrczc35bt-1',
    'data_augmentation':False,
    'data_aug_parameter': {'noise_factor': 0.2, 'nb_black_boxes': 10, 'rotation_max': 30, 'black_boxes_size': 8, 'resol': 1},
    'test_subject': 5,
}
dataset_path = cfg['dataset_path']
target_path = join(dataset_path, join('Subject1','Subject1.txt'))

print('[INFO] Test starting')

# Creating the path of the image
color_path = join(dataset_path, join('Subject1', join('Color','4510_color.png')))
depth_path = join(dataset_path, join('Subject1', join('Depth','4510_depth.png')))

# Loading images
rgb = cv2.imread(color_path)
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

# Searching the bounding boxes of the hand on the image
subject_target = np.loadtxt(target_path, dtype = str, delimiter=',')[1:,0:]
output = set_path_and_coord(color_path, depth_path, subject_target[0][27])

y0 = int(float(output[2]))
x0 = int(float(output[3]))
y1 = y0 + int(float(output[4]))
x1 = x0 + int(float(output[5]))

height = int(float(output[4]))
width = int(float(output[5]))

fig, ax = plt.subplots(1)
ax.imshow(rgb[x0:x1,y0:y1])

plt.show()

# Just keep the part of the image with the hand
rgb, depth = set_image(rgb, depth, x0, x1, y0, y1, width, height, True)

# Resize the image
rgb = np.asarray(cv2.resize(rgb, (100,100)))
depth = np.asarray(cv2.resize(depth, (100,100)))

# Concatenate the depth and the rgb images
image = np.concatenate((rgb, depth[..., None]), axis=2)

# Put the channels as the first dimension
image = np.transpose(image, (2, 0, 1))

# We normalize the image
image = image.astype(np.float32)
image[0:3,:] /= 255.0
image[3,:] /= 65535.0

image /= image.max()

# Convert the image in tensor
image = torch.tensor(image).to(torch.float)

print("[INFO] Test pour voir l'image de base")
image_rgb = image[0:3,:].clone()
image_rgb *= 255
image_rgb = np.transpose(image_rgb, (1,2,0))
image_rgb = image_rgb.numpy().astype(np.uint8)

fig, ax = plt.subplots(1)
ax.imshow(image_rgb)

plt.show()

print('[INFO] Test of gaussian_noise')
gaussian_image = gaussian_noise(image, cfg['data_aug_parameter']['noise_factor'], width, height)

gaussian_image_rgb = gaussian_image[0:3,:]
gaussian_image_rgb *= 255

gaussian_image_rgb = gaussian_image_rgb.numpy().astype(np.uint8)
gaussian_image_rgb = np.transpose(gaussian_image_rgb, (1,2,0))

fig, ax = plt.subplots(1)
ax.imshow(gaussian_image_rgb)

plt.show()

print('[INFO] Test of random_blocks')
random_blocks_image = random_blocks(image, cfg['data_aug_parameter']['nb_black_boxes'], cfg['data_aug_parameter']['black_boxes_size'])

random_blocks_image_rgb = random_blocks_image[0:3,:]
random_blocks_image_rgb *= 255
random_blocks_image_rgb = random_blocks_image_rgb.numpy().astype(np.uint8)
random_blocks_image_rgb = np.transpose(random_blocks_image_rgb, (1,2,0))

fig, ax = plt.subplots(1)
ax.imshow(random_blocks_image_rgb)

plt.show()

print('[INFO] Test of rotation')
rotation_image = rotation(image, cfg['data_aug_parameter']['rotation_max'])

rotation_image_rgb = rotation_image[0:3,:]
rotation_image_rgb *= 255
rotation_image_rgb = rotation_image_rgb.numpy().astype(np.uint8)
rotation_image_rgb = np.transpose(rotation_image_rgb, (1,2,0))

fig, ax = plt.subplots(1)
ax.imshow(rotation_image_rgb)

plt.show()

print('[INFO] Creation Dataloader')
train_dataset = hand_gesture_dataset.HGD(cfg['dataset_path'],
                                   depth=True,
                                   test=False, 
                                   transform=cfg['data_augmentation'],
                                   data_aug_parameter=cfg['data_aug_parameter'],
                                   test_subject=cfg['test_subject'])

train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, drop_last=True)

print('[INFO] Dataloader created')

print('[INFO] Test des images qui sont dans le dataloader')

images= next(iter(train_dataloader))['image']
images_rgb = images[0][0:3,:]
images_rgb *= 255
images_rgb = images_rgb.numpy().astype(np.uint8)

images_rgb = np.transpose(images_rgb, (1,2,0))

fig, ax = plt.subplots(1)
ax.imshow(images_rgb)

plt.show()

images_depth = images[0][3,:]
images_depth *= 65535
images_depth = images_depth.numpy().astype(np.uint16)

fig, ax = plt.subplots(1)
ax.imshow(images_depth)

plt.show()

print('[INFO] End Test')

#print('[INFO] ')