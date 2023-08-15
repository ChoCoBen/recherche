import os

import json
from os.path import join
from src.core import model
import torch
from torch.utils.data import DataLoader
from src import hand_gesture_dataset
from src.utils.losses import *
from src.utils.helper import to_device

results_dir = '/home/travail/Code/Results.json'
dataset_path = '/home/travail/Dataset/ndrczc35bt-1'
new_results_dir = '/home/travail/Code/Some_Results/evolution_resol.json'

param = {
    'data_augmentation': False,
    'data_aug_parameter': {'noise_factor': 0.2, 'nb_black_boxes': 10, 'rotation_max': 30, 'black_boxes_size': 8, 'resol': 1},
    'learning_rate': 0.0001,
    'dropout': 0.3,
    'depth': True,
}

seed = 0
name = 'classic_noDataAugmentation_noDepth_resol20'

subjects = ['subject1', 'subject2', 'subject3', 'subject4', 'subject5']
checks_dir = []

with open(results_dir, "r") as file:
    file = json.load(file)
    
for result in file:
    # We search the corresponding object
    if seed == result['seed'] and name == result['name'] and param['data_augmentation'] == result['param']['data_augmentation']:
        if (param['data_augmentation'] and param == result['param']) or not param['data_augmentation']:
            for subject in subjects:
                checks_dir.append(result['results'][subject]['check_dir'])

if len(checks_dir) > 5:
    print('[WARNING] 2 networks are found with these parameters')

gathered_result = {}
subject = 0
resols = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for check_dir in checks_dir:
    subject += 1
    files_list = os.listdir(check_dir)

    network_file = [file for file in files_list if not file.endswith('.json')][0]
    mod = model.ConvModel({'dropout': 0.3, 'learning_rate': 0.0001})
    mod.load_state_dict(torch.load(join(check_dir, network_file))['CONV'])
    mod.to('cuda')
    mod.eval()

    for resol in resols:
        param['data_aug_parameter']['resol'] = resol
        test_dataloader = hand_gesture_dataset.HGD(dataset_path,
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
            gathered_result.setdefault(resol, {}).setdefault(subject, average_losses(loss_history)['test_accuracy'][0])

with open(new_results_dir, "r") as file:
    new_res = json.load(file)

new_res.setdefault(name, gathered_result)

with open(new_results_dir, "w") as file:
    json.dump(new_res, file)