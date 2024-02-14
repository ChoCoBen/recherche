from torch.utils.data import Dataset
import os
from os.path import join
import numpy as np
import cv2
import torch
from src.utils.BButils.BBdataset_helper import set_coord

from src.utils.cobotic_helper import set_image

class CHD(Dataset):
    """This class represents the Hand bounding boxes detection dataset and is intended for use with torch.Dataloader"""
    def __init__(self, path_to_dataset: str,
                test: bool=False,
                test_subject: int=5,
                cfg: dict={'depth':False}) -> None:
        """
        :param path_to_dataset: str Path to the folder where the dataset is saved
        :param test: bool Determine if we use the test set or the training set
        :param test_subject: int Determine wich subject we want as being the test subject
        """
        super().__init__()
        self.path_to_dataset = path_to_dataset
        self.cfg = cfg
        self.gi_to_i = {'g1':1, 'g2':2, 'g3':3, 'g4':4, 'g5':5, 'g6':6, 'g7':7, 'g8':8, 'g9':9, 'g10':1, 'g11':2, 'g12':3, 'g13':4, 'g14':5, 'g15':6, 'g16':10, 'g17':10}
        subjects = ['u'+str(i) for i in range (1, 11)]
        subject_to_subjects = {1:(subjects[2:10],subjects[0:2]), 2:(subjects[0:2]+subjects[4:10],subjects[2:4]), 3:(subjects[0:4]+subjects[6:10],subjects[4:6]), 4:(subjects[0:6]+subjects[8:10],subjects[6:8]), 5:(subjects[0:8],subjects[8:10])}
        train_subjects, test_subjects = subject_to_subjects[test_subject]

        subjects = test_subjects if test else train_subjects

        targets_indices, targets, images = [], [], []
        for ui in subjects:
            subject_path = join(self.path_to_dataset, ui)
            for gi in ['g'+str(i) for i in range (1,18)]:
                gi_path = join(subject_path, gi)
                for ci in ['c'+str(i) for i in range (1,7)]:
                    ci_path = join(gi_path, ci)
                    if 'annotation' in os.listdir(ci_path):
                        annotations = os.listdir(join(ci_path, 'annotation'))
                        annotations.remove('classes.txt')
                        
                        for annotation in annotations:
                            annotation_path = join(join(ci_path, 'annotation'), annotation)
                            with open(annotation_path, 'r') as file:
                                bounding_box = file.readline().strip().split()
                                bounding_box = [float(nombre) for nombre in bounding_box[-4:]]

                                color_path = join(join(ci_path, 'color'), annotation.replace('txt', 'png'))
                                depth_path = join(join(ci_path, 'depth'), annotation.replace('txt', 'png'))
                                
                                target = np.zeros(10)
                                target[self.gi_to_i[gi]-1] = 1

                                targets_indices.append(self.gi_to_i[gi]-1)
                                targets.append(target)
                                images.append(np.array([color_path] + [depth_path] + bounding_box))
        
        self.output = {'images': np.array(images, dtype=object), 'targets':np.array(targets), 'targets_indices':np.array(targets_indices)}

    def __getitem__(self, idx) -> torch.Tensor:
        output = {'image': None, 'target': None, 'target_indice': None}
        rgb = cv2.imread(self.output['images'][idx][0])
        depth = cv2.imread(self.output['images'][idx][1], cv2.IMREAD_UNCHANGED)
        use_depth = self.cfg['depth']

        total_height, total_width = rgb.shape[:2]

        BB = self.output['images'][idx][2:]
        
        coord = [int((float(BB[0])-float(BB[2])/2)*total_width),
         int((float(BB[1])-float(BB[3])/2)*total_height),
         int((float(BB[0])+float(BB[2])/2)*total_width),
         int((float(BB[1])+float(BB[3])/2)*total_height)]

        rgb, depth = set_image(rgb, depth, coord, use_depth)
        
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

        output['image'] = image
        output['target'] = torch.tensor(self.output['targets'][idx]).to(torch.float)
        output['target_indice'] = torch.tensor(self.output['targets_indices'][idx]).to(torch.uint8)

        return output

    def __len__(self) -> int:
        return self.output['targets_indices'].size