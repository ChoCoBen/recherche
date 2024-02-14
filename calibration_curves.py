import torch
from src import cobotic_dataset
from src.utils.helper import load_config, to_device
from src.core import model
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

PATHS ={
    1: "/store/travail/checkpoint/run_2023-11-06_20h51m49s/classic_cobotic_10s_v2_seed0_subject1",
    2: "/store/travail/checkpoint/run_2023-11-04_01h02m35s/classic_cobotic_10s_v2_seed0_subject2",
    3: "/store/travail/checkpoint/run_2023-11-03_15h46m57s/classic_cobotic_10s_v2_seed0_subject3",
    4: "/store/travail/checkpoint/run_2023-10-31_17h21m51s/classic_cobotic_10s_v2_seed0_subject4",
    5: "/store/travail/checkpoint/run_2023-10-30_22h50m54s/classic_cobotic_10s_v2_seed0_subject5"
}

def load_mod_cfg_data(test_subject):
    cfg = "./configs/train.yaml"
    cfg = load_config(cfg)

    ## Load and setup the model
    PATH = PATHS[test_subject]
    cfg['num_classes'] = 10

    mod = model.ConvModel(cfg)
    mod.load_state_dict(torch.load(PATH)['CONV'])
    mod.to('cuda')
    mod.eval()

    dataset_path = '/store/travail/data_sorted'

    dataset = cobotic_dataset.CHD(dataset_path, test=True, test_subject=test_subject)
    dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=False, drop_last=True)

    return cfg, mod, dataloader

def test_model_on_data(mod, dataloader):
    nb_pred = {i/20:0 for i in range(20)}
    nb_correct_pred = {i/20:0 for i in range(20)}
    with torch.no_grad():
        for batch in dataloader:
            batch = to_device(batch, 'cuda')
            target = batch['target_indice']
            image = batch['image']

            predictions = mod(image)
            predictions = nn.functional.softmax(predictions, dim=1)

            predictions = predictions.to('cpu').tolist()
            target = target.to('cpu').tolist()

            for num_pred, prediction in enumerate(predictions):
                correct_proba = prediction[target[num_pred]]
                for i in range(20):
                        if i/20 <= correct_proba < (i+1)/20:
                            nb_correct_pred[i/20] += 1
                            break
                for proba in prediction:
                    for i in range(20):
                        if i/20 <= proba < (i+1)/20:
                            nb_pred[i/20] += 1
                            break

    percentage_correct_pred = {k:nb_correct_pred[k]/nb_pred[k] for k in nb_pred.keys()}
    return percentage_correct_pred

def plot_calibration_curves(percentage_correct_pred):
    x = list(percentage_correct_pred.keys())
    y = list(percentage_correct_pred.values())
    plt.plot(x, x, label='perfect calibration')
    plt.plot(x, y, label='model calibration')
    plt.xlabel('predicted probability')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

def plot_all_calibration_curves(percentages_correct_pred_list):
    # draw every curves on the same plot
    P_LIST = ['P1-P2', 'P3-P4', 'P5-P6', 'P7-P8', 'P9-P10']
    for i, percentages_correct_pred in enumerate(percentages_correct_pred_list):
        x = list(percentages_correct_pred.keys())
        y = list(percentages_correct_pred.values())
        plt.plot(x, y, label=f'modèle {P_LIST[i]}')
        plt.xlabel('Probabilité prédite')
        plt.ylabel('Fréquence réelle des évènements')
    plt.plot(x, x, label='calibration parfaite')
    plt.legend()
    plt.show()


def plot_all():
    percentage_correct_pred_list = []
    for i in range(1,6):
        cfg, mod, dataloader = load_mod_cfg_data(i)
        percentage_correct_pred_list.append(test_model_on_data(mod, dataloader))
        plot_calibration_curves(percentage_correct_pred_list[i-1])
    
    plot_all_calibration_curves(percentage_correct_pred_list)

plot_all()
    
