import json
import matplotlib.pyplot as plt
from os.path import join
import os

results_dir = '/home/travail/Code/Results.json'

param = {
    'data_augmentation': False,
    'data_aug_parameter': {'noise_factor': 0.2, 'nb_black_boxes': 10, 'rotation_max': 30, 'black_boxes_size': 8, 'resol': 1},
    'learning_rate': 0.0001,
    'dropout': 0.3,
    'depth': True,
}

seed = 0
name = 'classic_noDataAugmentation_noDepth_resol100'

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

subject = 0
for check_dir in checks_dir:
    subject += 1
    if "no_folder" != check_dir:
        result_file = join(check_dir, 'result_history.json')
        if os.path.exists(result_file):
            with open(result_file, "r") as file:
                valid_history = json.load(file)
            
            acc = []
            for valid_result in valid_history:
                acc.append(valid_result['test_accuracy'][0])
            
            x = range(len(acc))

            plt.plot(x, acc, label='lr='+str(param['learning_rate']))#'subject '+str(subject))

            max_y = max(acc)
            plt.annotate(f'Max: {round(max_y,2)}', xy=(0, max_y), xytext=(20, 0), textcoords='offset points')

            plt.ylim(ymin=0)
            plt.legend()
            plt.title("Résultats de validation lors de l'entraînement")
            plt.xlabel('Accuracy')
            plt.ylabel('Numéro du test')
            plt.show()
        else: print("[Warning] There is a network that does not have loss history registererd")


        
