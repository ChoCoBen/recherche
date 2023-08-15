import json
from src.core import model
import torch
from torch.utils.data import DataLoader
from src import hand_gesture_dataset
import os
from os.path import join
from src.utils.losses import *
from src.utils.helper import to_device

results_dir = '/home/travail/Code/Results.json'
dataset_path = '/home/travail/Dataset/ndrczc35bt-1'

param = {
    'data_augmentation': False,
    'data_aug_parameter': {'noise_factor': 0.2, 'nb_black_boxes': 10, 'rotation_max': 30, 'black_boxes_size': 8, 'resol': 1},
    'learning_rate': 0.0001,
    'dropout': 0.3,
    'depth': True,
}

seed = 0
name = 'classic_noDataAugmentation_withDepth_resol100'

subjects = ['subject1', 'subject2', 'subject3', 'subject4', 'subject5']
checks_dir = []

with open(results_dir, "r") as file:
    file = json.load(file)

acc_mean = []
for result in file:
    # We search the corresponding object
    if seed == result['seed'] and name == result['name'] and param['data_augmentation'] == result['param']['data_augmentation']:
        if (param['data_augmentation'] and param == result['param']) or not param['data_augmentation']:
            for subject in subjects:
                checks_dir.append(result['results'][subject]['check_dir'])
                acc_mean.append(result['results'][subject]['acc_mean'])

if len(checks_dir) > 5:
    print('[WARNING] 2 networks are found with these parameters')

subject = 0
acc_mean_no_depth = []
for check_dir in checks_dir:
    subject += 1
    files_list = os.listdir(check_dir)

    network_file = [file for file in files_list if not file.endswith('.json')][0]
    mod = model.ConvModel({'dropout': 0.3, 'learning_rate': 0.0001})
    mod.load_state_dict(torch.load(join(check_dir, network_file))['CONV'])
    mod.to('cuda')
    mod.eval()

    test_dataloader = hand_gesture_dataset.HGD(dataset_path,
                                    depth=False,
                                    test=True,
                                    data_aug_parameter=param['data_aug_parameter'],
                                    test_subject=subject)
        
    test_dataloader = DataLoader(dataset=test_dataloader, batch_size=10, shuffle=True, drop_last=True)

    with torch.no_grad():
        loss_history = None
        for batch in test_dataloader:
            batch = to_device(batch, 'cuda')
            image = batch['image']
            prediction = mod(image)
            prediction = torch.sigmoid(prediction)
            losses = compute_losses(prediction, batch['target_indice'], test= True)
            loss_history = append_losses(losses, loss_history)
        
        loss_history['total_loss_test'] = to_device(loss_history['total_loss_test'], 'cpu')
        loss_history['test_confidence'] = to_device(loss_history['test_confidence'], 'cpu')

        acc_mean_no_depth.append(average_losses(loss_history)['test_accuracy'][0])

acc = {
    'acc_mean':acc_mean,
    'acc_mean_no_depth':acc_mean_no_depth
}

print(acc)