from torch.utils.data import Dataset
import os
from os.path import join
import numpy as np
import cv2
import torch
from src.utils.BButils import BBdataset_helper
from src.utils.dataset_helper import reduce_resolution

class HBBD(Dataset):
    """This class represents the Hand bounding boxes detection dataset and is intended for use with torch.Dataloader"""
    def __init__(self, path_to_dataset: str,
                test: bool=False,
                test_subject: int=5,
                cfg: dict={}) -> None:
        """
        :param path_to_dataset: str Path to the folder where the dataset is saved
        :param test: bool Determine if we use the test set or the training set
        :param test_subject: int Determine wich subject we want as being the test subject
        """
        super().__init__()
        self.path_to_dataset = path_to_dataset
        self.resol = cfg['resol'] if 'resol' in cfg else 1

        train_subjects = os.listdir(self.path_to_dataset)
        train_subjects.sort()

        test_subjects = train_subjects.pop(test_subject-1)

        subjects = [test_subjects] if test else train_subjects

        self.number_of_images = 1950 * (1 - (test - 1) * 3) # 2400 images by subjects but some of them contain two different gestures made by the two hands, just 1950 contain single hand gestures

        self.output = {'images': np.zeros((self.number_of_images, 2), dtype=object), 'targets':np.zeros((self.number_of_images, 2, 4)), 'targets_gesture': np.zeros((self.number_of_images, 2))}        

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

                coords, gestures = [], []
                for gesture in range(subject_target.shape[1]):
                    if subject_target[idx, gesture] != '[0 0 0 0]' and gesture > 1:
                        coord = np.zeros(4)
                        coord[:] = subject_target[idx, gesture][1:len(subject_target[idx, gesture])-1].split()
                        # coord is y1, x1, height, width -> we want x1, y1, x2, y2
                        coord = BBdataset_helper.set_coord(coord)
                        coords.append(coord)
                        gestures.append(gesture - 2)
                
                if len(coords) == 2:
                    self.output['targets'][count_subject + count_image, 0] = coords[0]
                    self.output['targets'][count_subject + count_image, 1] = coords[1]
                    self.output['images'][count_subject + count_image][0] = rgb_image_path
                    self.output['images'][count_subject + count_image][1] = depth_image_path
                    self.output['targets_gesture'][count_subject + count_image] = np.array(gestures)

                    count_image += 1
            count_subject += 1950 # 1950 images with single hand gestures

    def __getitem__(self, idx) -> torch.Tensor:
        """Fetches an image of a hand gesture
        
        :param index: int index of the image
        :return: torch.Tensor a the corresponding image
        """
        output = {'image': None, 'First_hand': None, 'Second_hand': None, 'First_gesture': None, 'Second_gesture': None}
        image = cv2.imread(self.output['images'][idx][0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.resol != 1:
            image = reduce_resolution(image, self.resol)

        output['image'] = image
        output['First_hand'] = self.output['targets'][idx][0]
        output['Second_hand'] = self.output['targets'][idx][1]
        output['First_gesture'] = self.output['targets_gesture'][idx][0]
        output['Second_gesture'] = self.output['targets_gesture'][idx][1]

        return output

    def __len__(self) -> int: 
        """Outputs the total number of images in the dataset
        
        :return: int Number of images
        """
        return  self.number_of_images