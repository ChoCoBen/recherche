from torch.utils.data import Dataset
import os
from os.path import join
import numpy as np
import cv2
from src.utils.dataset_helper import *
from src.utils.data_augmentation import *
import torch
import random

class HGD(Dataset):
    """This class represents the Hand Gesture dataset and is intended for use with torch.Dataloader"""

    def __init__(self, path_to_dataset: str,
                 depth: bool,
                 test: bool=False,
                 transform: bool=False,
                 data_aug_parameter: dict=None,
                 test_subject: int=5) -> None:
        """
        :param path_to_dataset: str Path to the folder where the dataset is saved
        :param test: bool Determine if we use the test set or the training set
        :param depth: bool Determine if we use the depth or set the depth to 0
        :param transform: bool Determine if we use data augmentation
        :param data_aug_parameter: dict Contains parameters for data augmentation
        :param test_subject: int Determine wich subject we want as being the test subject
        """ 
        super().__init__()
        self.path_to_dataset = path_to_dataset
        self.transform = transform
        self.data_augmentation_parameter = data_aug_parameter
        self.depth = depth
        ## 4 subjects are used for training, 1 for testing

        train_subjects = os.listdir(self.path_to_dataset)
        train_subjects.sort()

        test_subjects = train_subjects.pop(test_subject-1)

        if test: subjects = [test_subjects]
        else: subjects = train_subjects

        self.number_of_images = 2400 * (1 - (test - 1) * 3) # 2400 images by subjects but some of them contain two different gestures made by the two hands
        self.number_of_gestures = 4350 * (1 - (test - 1) * 3)
        self.output = {'images': np.zeros((self.number_of_gestures, 6), dtype=object), 'targets':np.zeros((self.number_of_gestures, 29)), 'targets_indices': np.zeros((self.number_of_gestures))}

        ## Load the images and the targets
        count_subject = 0
        for subject in subjects:

            # Set every path
            subject_path = join(self.path_to_dataset, subject)
            rgb_path = join(subject_path, 'Color')
            depth_path = join(subject_path, 'Depth')
            target_path = join(subject_path, subject +'.txt')

            # Load targets
            subject_target = np.loadtxt(target_path, dtype = str, delimiter=',')[1:,0:]

            # Load images
            count_image = 0
            for idx in range(subject_target.shape[0]):
                image_num = subject_target[idx, 0].split('\\')[3].split('_')[0]
                rgb_image_path = join(rgb_path, image_num + '_color.png')
                depth_image_path = join(depth_path, image_num + '_depth.png')

                for gesture in range(subject_target.shape[1]):
                    if subject_target[idx, gesture] != '[0 0 0 0]' and gesture > 1:
                        self.output['targets_indices'][count_subject + count_image] = gesture - 2
                        self.output['targets'][count_subject + count_image] = set_target(gesture - 2)
                        self.output['images'][count_subject + count_image] = set_path_and_coord(rgb_image_path, depth_image_path, subject_target[idx, gesture])
                        count_image += 1
            
            count_subject += 4350 # 4350 hands gestures by subjects
    
    def __getitem__(self, idx) -> torch.Tensor:
        """Fetches an image of a hand gesture
        
        :param index: int index of the image
        :return: torch.Tensor a the corresponding image
        """
        output = {'image': None, 'target': None, 'target_indice': None}
        rgb = cv2.imread(self.output['images'][idx][0])
        depth = cv2.imread(self.output['images'][idx][1], cv2.IMREAD_UNCHANGED)
        use_depth = self.depth

        y1 = int(float(self.output['images'][idx][2]))
        x1 = int(float(self.output['images'][idx][3]))
        
        height = int(float(self.output['images'][idx][4]))
        width = int(float(self.output['images'][idx][5]))
        
        y2 = y1 + height
        x2 = x1 + width

        # Just keep the part of the image with the hand
        rgb, depth = set_image(rgb, depth, x1, x2, y1, y2, width, height, use_depth)

        resol = self.data_augmentation_parameter['resol']
        if resol != 1:
            rgb, depth = reduce_resolution(rgb, resol), reduce_resolution(depth, resol)

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

        # Use data augmentation if wanted
        if self.transform:
            image = self.aug(image, width, height)

        # Stock the tensors in the output
        output['image'] = image
        output['target'] = torch.tensor(self.output['targets'][idx]).to(torch.float)
        output['target_indice'] = torch.tensor(self.output['targets_indices'][idx]).to(torch.uint8)

        return output
    
    def __len__(self) -> int: 
        """Outputs the total number of gestures in the dataset
        
        :return: int Number of gestures
        """
        return  self.number_of_gestures
    
    def aug(self, image, width, height):
        do_gaussian = random.choice([True, False, False])
        do_rotation = random.choice([True, False, False])
        do_random_blocks = random.choice([True, False, False])

        noise_factor = self.data_augmentation_parameter['noise_factor']
        nb_black_boxes = self.data_augmentation_parameter['nb_black_boxes']
        black_boxes_size = self.data_augmentation_parameter['black_boxes_size']
        rotation_max = self.data_augmentation_parameter ['rotation_max']

        if do_gaussian:
            image = gaussian_noise(image, noise_factor, width, height)
        
        if do_random_blocks:
            image = random_blocks(image, nb_black_boxes, black_boxes_size)
        
        if do_rotation:
            image = rotation(image, rotation_max)

        return image