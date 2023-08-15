import torch
import yaml
import os
import shutil
from torch.utils.data import DataLoader
from src import hand_gesture_dataset
import wandb
import json
from datetime import datetime

def to_device(tensors, device):
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, dict):
        return dict(
            (key, to_device(tensor, device)) for (key, tensor) in tensors.items()
        )
    elif isinstance(tensors, list):
        return list(
            (to_device(tensors[0], device), to_device(tensors[1], device))
        )
    else: raise NotImplementedError("Unknown type {0}".format(type(tensors)))

def load_config(path: str) -> dict:
    """Load config from YAML file
    
    :param path: str Path to the file
    :return: dict Dict of the config
    """
    with open(path, 'r') as file:
        cfg = yaml.safe_load(file)
    
    return cfg

def setup_device(cfg: dict):
    device = cfg['device']
    if device == 'cpu':
        device = torch.device('cpu')
        print('[INFO] Running on CPU.')
    elif device == 'cuda':
        if not torch.cuda.is_available():
            raise Exception('[FATAL] Device set to "gpu" in config file but CUDA is not available.')
        os.environ['CUDA_AVAILABLE_DEVICES'] = cfg['gpu_num']
        device = torch.device('cuda:0')
        print('[INFO] Running on GPU.')
    else: raise Exception(f'[FATAL] Device type "{device}"not supported.')

    return device

def setup_dataloaders(cfg):
    assert os.path.exists(cfg['dataset_path'])
    datasetObject = getattr(hand_gesture_dataset, cfg['dataset'])
    print('[INFO] Creating dataloader')

    train_dataset = datasetObject(cfg['dataset_path'],
                                   depth=cfg['depth'],
                                   test=False, 
                                   transform=cfg['data_augmentation'],
                                   data_aug_parameter=cfg['data_aug_parameter'],
                                   test_subject=cfg['test_subject'])
    
    test_dataset = datasetObject(cfg['dataset_path'],
                                 depth=cfg['depth'],
                                 test=True,
                                 transform=False,
                                 data_aug_parameter=cfg['data_aug_parameter'],
                                 test_subject=cfg['test_subject'])
                                 

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)
    print('[Dataloader OK]')

    return train_dataloader, test_dataloader

def setup_wandb(cfg, wandb_cfg, model):
    wandb.init(project=wandb_cfg['project_name'], group=wandb_cfg['group'], entity=wandb_cfg['entity'], config=cfg)

    if wandb_cfg['watch_conv']:
        print('[INFO] Logging convolutional network info')
        wandb.watch(model, log='all', idx=1, log_freq=wandb_cfg['watch_every'], log_graph=False)
    
    print('[INFO] WandB initialized')

def print_time(time: float):
    time = int(time)
    seconds = str(time % 60).zfill(2)
    time = time // 60
    minutes = str(time % 60).zfill(2)
    hours = str(time // 60).zfill(2)
    return f'{hours}:{minutes}:{seconds}'

def save_result_model(current_result, model, cfg):
    param, seed, name, subject = set_param_seed_name(cfg)

    results_dir = cfg['results_dir']

    current_check_dir = os.path.join(cfg['checkpoint_dir'], 'run_'+datetime.today().strftime('%Y-%m-%d_%Hh%Mm%Ss'))

    with open(results_dir, "r") as file:
        results = json.load(file)
    
    isexisting = False
    isbest = False

    for result in results:
        # First, we check if the model is already in the results
        if seed == result['seed'] and name == result['name'] and param['data_augmentation'] == result['param']['data_augmentation']:
            if (param['data_augmentation'] and param == result['param']) or not param['data_augmentation']:
                isexisting = True

                # Then, we check if the results are better
                subject_result = result['results'][subject]
                if subject_result['acc_mean'] < current_result['test_accuracy'][0]:
                    
                    # We change the results if they are better
                    subject_result['acc_mean'] = float(current_result['test_accuracy'][0])
                    subject_result['acc_std'] = float(current_result['test_accuracy'][1])
                    subject_result['cfd'] = float(current_result['test_confidence'][0])

                    ## We also change the model corresponding to these new results
                    if os.path.exists(subject_result['check_dir']):
                        shutil.rmtree(subject_result['check_dir'])

                    subject_result['check_dir'] = current_check_dir

                    if not os.path.exists(current_check_dir):
                        os.makedirs(current_check_dir)
                    checkpoint_name = f'{name}_seed{seed}_{subject}'
                    torch.save({'CONV': model.state_dict(), 'optim': model.optim.state_dict()}, os.path.join(current_check_dir, checkpoint_name))
                    isbest = True

    if not isexisting:
        # If the model is not already in the results, then we create a new result object for him
        default_result_subject = { 
            'acc_mean': 0,
            'acc_std': 0,
            'cfd': 0,
            'check_dir':'no_folder',
        }

        # No need to set the values of accuracies of the current_result, if it doesn't exist, it means that it the first time we test this model, the results are going to be better in the future anyway
        default_result_subjects = {
            'subject1': default_result_subject,
            'subject2': default_result_subject,
            'subject3': default_result_subject,
            'subject4': default_result_subject,
            'subject5': default_result_subject,
        }

        default_result = {
            'name': name,
            'seed': seed,
            'param': param,
            'results': default_result_subjects,
        }

        results.append(default_result)
        isbest = True

    with open(results_dir, "w") as file:
        json.dump(results, file)

    return isbest

def save_history(result_history, cfg):
    param, seed, name, subject = set_param_seed_name(cfg)

    results_dir = cfg['results_dir']

    # We searcg where we can write the history of results
    with open(results_dir, "r") as file:
        file = json.load(file)
    
    for result in file:
        # We search the corresponding object
        if seed == result['seed'] and name == result['name'] and param['data_augmentation'] == result['param']['data_augmentation']:
            if (param['data_augmentation'] and param == result['param']) or not param['data_augmentation']:
                check_dir = result['results'][subject]['check_dir']
                break
    
    with open(os.path.join(check_dir, 'result_history.json'), "w") as history_result_file:
        json.dump(result_history, history_result_file)

def set_param_seed_name(cfg):
    param = {
        'data_augmentation': cfg['data_augmentation'],
        'data_aug_parameter': cfg['data_aug_parameter'],
        'learning_rate': cfg['learning_rate'],
        'dropout': cfg['dropout'],
        'depth': cfg['depth'],
    }

    seed = cfg['seed']
    name = cfg['name']
    subject = f"subject{cfg['test_subject']}"

    return param, seed, name, subject