import matplotlib
from src import hand_gesture_dataset
import torch
from os.path import join
import numpy as np
from src.utils.dataset_helper import set_path_and_coord
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from skimage import io
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import torchvision.transforms as T

print('[INFO] Test starting')

dataset = 'HGD'
dataset_path = '/home/travail/Dataset/ndrczc35bt-1'
color_path = join(dataset_path, join('Subject1', join('Color','4510_color.png')))
depth_path = join(dataset_path, join('Subject1', join('Depth','4510_depth.png')))
target_path = join(dataset_path, join('Subject1','Subject1.txt'))

subject_target = np.loadtxt(target_path, dtype = str, delimiter=',')[1:,0:]
output = set_path_and_coord(color_path, depth_path, subject_target[0][24])

y0 = int(float(output[2]))
x0 = int(float(output[3]))
y1 = y0 + int(float(output[4]))
x1 = x0 + int(float(output[5]))

rect1 = patches.Rectangle((x0, y0), (x1-x0), (y1-y0), linewidth=1, facecolor='none')
rect2 = patches.Rectangle((x0, y0), (x1-x0), (y1-y0), linewidth=1, facecolor='none')
img_color = np.array(io.imread(color_path))
img_depth = np.array(io.imread(depth_path))

img_color_cut = img_color[x0:x1, y0:y1]
img_depth_cut = img_depth[x0:x1, y0:y1]

print('[INFO] Data MEGURU')

fig, ax = plt.subplots(1)
ax.imshow(img_color)
ax.add_patch(rect1)

fig, ax = plt.subplots(1)
ax.imshow(img_depth)
ax.add_patch(rect2)

fig, ax = plt.subplots(1)
ax.imshow(img_color_cut)

fig, ax = plt.subplots(1)
ax.imshow(img_depth_cut)

plt.show()

print(np.min(img_depth))
print(np.max(img_depth))

print(np.min(img_depth_cut))
print(np.max(img_depth_cut))

print('[INFO] Data perso')

path_to_rgb = join(join("../../Dataset/PersonalData", "color_select"), "000386.jpg")
path_to_depth = join(join("../../Dataset/PersonalData", "depth_select"), "000386.jpg")
path_to_depth = join(join("../../Dataset/PersonalData", "depth"), "000386.png")

path_to_rgb_cut = join(join("../../Dataset/PersonalData", "color_select"), "03_proche_color.jpg")
path_to_depth_cut = join(join("../../Dataset/PersonalData", "depth_select"), "03_proche_depth.jpg")

img_color = np.array(io.imread(path_to_rgb))
img_depth = np.array(io.imread(path_to_depth))

print(type(img_depth[0,0]))

img_color_cut = np.array(io.imread(path_to_rgb_cut))
img_depth_cut = np.array(io.imread(path_to_depth_cut))

fig, ax = plt.subplots(1)
ax.imshow(img_color)

fig, ax = plt.subplots(1)
ax.imshow(img_depth)

fig, ax = plt.subplots(1)
ax.imshow(img_color_cut)

fig, ax = plt.subplots(1)
ax.imshow(img_depth_cut)

plt.show()

print(np.min(img_depth))
print(np.max(img_depth))

print(np.min(img_depth_cut))
print(np.max(img_depth_cut))

print('[INFO] Test for preprocessing method of resnet')

weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()

# Apply it to the input image
img = Image.open(depth_path)
resize_size = 232
crop_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
toTensor = T.Compose([T.ToTensor()])
transform = T.Compose([
    T.Resize(resize_size, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(crop_size),
    T.Normalize(mean=mean, std=std)
])
img_tensor = toTensor(img).to(torch.float32)
img_tensor_rescaled = (img_tensor-torch.min(img_tensor))/(torch.max(img_tensor)-torch.min(img_tensor))
print(torch.max(img_tensor_rescaled))
img_tensor_rescaled_dup = img_tensor_rescaled.repeat(3, 1, 1)
img_transformed = transform(img_tensor_rescaled_dup)
# img_transformed = preprocess(img_tensor)

image_np = img_transformed.numpy()
fig, ax = plt.subplots(1)
ax.imshow(image_np.transpose(1, 2, 0))

print(torch.max(img_transformed))

print('[INFO] End Test')
