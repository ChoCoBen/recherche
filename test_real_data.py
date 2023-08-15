import cv2
from os.path import join, exists
from src.core import model
import numpy as np
import torch
from src.utils.helper import *

## Load and setup the images
path_to_images = "../../Dataset/PersonalData"

path_to_rgbs = join(path_to_images, "color_select")
path_to_depths = join(path_to_images, "depth_select")

path_to_rgb = join(path_to_rgbs, "03_proche_color.jpg")
path_to_depth = join(path_to_depths, "03_proche_depth.jpg")

rgb = cv2.imread(path_to_rgb)
depth = cv2.imread(path_to_depth)

depth = depth[:,:,1]

rgb = np.asarray(cv2.resize(rgb, (100,100)))
depth = np.asarray(cv2.resize(depth, (100,100)))

image = np.concatenate((rgb, depth[..., None]), axis=2)
image = np.transpose(image, (2, 0, 1))

image = torch.tensor(image).to(torch.float)

## Load and setup the model
PATH = "../LSTM2/checkpoint/run_2022-11-28_15h22m03s/epoch14_batch26100.tar"
cfg = "./configs/train.yaml"
cfg = load_config(cfg)

mod = model.ConvModel(cfg)
mod.load_state_dict(torch.load(PATH)['CONV'])
mod.eval()

## Testing on the image
prediction = mod(image.view(1, 4, 100, 100))
prediction = torch.sigmoid(prediction)
indice_pred = torch.argmax(prediction)
print(indice_pred)

dict_gesture = {0:'poing droite', 1:'poing gauche', 2:'1 droite', 3:'1 gauche', 4:'2 droite', 5:'2 gauche', 6:'3 droite', 7:'3 gauche', 8:'4 droite', 9:'4 gauche', 10:'5 droite', 11:'5 gauche', 12:'6 droite', 13:'6 gauche', 14:'7 droite', 15:'7 gauche', 16:'8 droite', 17:'8 gauche', 18:'9 droite', 19:'9 gauche', 20:'geste non enregistré', 21:'geste non enregistré', 22:'geste non enregistré', 23:'geste non enregistré', 24:'geste non enregistré', 25:'geste non enregistré', 26:'geste non enregistré', 27:'geste non enregistré', 28:'geste non enregistré'}
print(dict_gesture[indice_pred.item()])