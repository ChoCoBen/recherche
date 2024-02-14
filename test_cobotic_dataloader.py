import cv2
from src import cobotic_dataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from src.utils.helper import setup_dataloaders

cfg = {'depth': True}
dataset_path = '/store/travail/data_sorted'

print('[INFO] Creation Dataloader')

dataset = cobotic_dataset.CHD(dataset_path,test_subject=3)
train_dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=True)


# rgb = cv2.imread(dataset.output['images'][0][0])
# total_height, total_width = rgb.shape[:2]

# mean = 0
# for i in range(dataset.__len__()):
#     BB = dataset.output['images'][i][4:]
#     mean += float(BB[0])*float(BB[1])*total_height*total_width
# mean /= dataset.__len__()
# print(f'Moyenne des BB: {mean}')
# 2848.6 vs 5709.9 pour HANDS

print(f'Taille du dataset: {dataset.__len__()}')

print('[INFO] Dataloader Created')

print('[INFO] Test des images qui sont dans le dataloader')
images= next(iter(train_dataloader))['image']

images_rgb = images[0][0:3,:]
images_rgb *= 255
images_rgb = images_rgb.numpy().astype(np.uint8)

images_rgb = np.transpose(images_rgb, (1,2,0))

fig, ax = plt.subplots(1)
ax.imshow(images_rgb)

plt.show()

print('[INFO] Test du parcours des données')

cfg = {'dataset_path': '/store/travail/data_sorted', 'dataset': 'CHD', 'depth': True, 'batch_size': 10, 'test_subject': 3}
train_dataloader, test_dataloader = setup_dataloaders(cfg)

for batch in train_dataloader:
    pass

for batch in test_dataloader:
    pass

print('[INFO] Fin du parcours des données')