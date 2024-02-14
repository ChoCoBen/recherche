from src.core import model
import torch
from src.utils.helper import *
from src.utils.losses import *

dict_gesture = {0:6,
                1:6,
                2:1,
                3:1,
                4:2,
                5:2,
                6:3,
                7:3,
                8:4,
                9:4,
                10:5,
                11:5,
                12:100,
                13:100,
                14:100,
                15:100,
                16:100,
                17:100,
                18:100,
                19:100,
                20:100,
                21:100,
                22:100,
                23:100,
                24:100,
                25:100,
                26:7,
                27:100,
                28:100}

def test_model_on_data(model, data, model_type):
    with torch.no_grad():
        acc = 0
        for batch in data:
            batch = to_device(batch, 'cuda')
            target = batch['target_indice']
            image = batch['image']

            prediction = model(image)
            prediction = torch.sigmoid(prediction) if model_type == 'Hands' else prediction
            prediction = nn.functional.softmax(prediction, dim=1)

            gesture_pred = torch.argmax(prediction, dim=1)
            if model_type == 'Hands':
                for i in range(len(gesture_pred)):
                    gesture_pred[i] = dict_gesture[gesture_pred[i].item()]-1
                acc_batch = torch.sum(gesture_pred == target).item() / target.shape[0]
                acc += acc_batch
            elif model_type == 'Cobotic':
                nb_correct_batch = 0
                for i in range(len(gesture_pred)):
                    for k,v in dict_gesture.items():
                        if v == gesture_pred[i].item()+1:
                            nb_correct_batch += 1 if k == target[i].item() else 0
                            
                acc_batch = nb_correct_batch/cfg['batch_size']
                acc += acc_batch

        # Compute the final losses
        return acc

print("[INFO] Test of Hands model on cobotic data")

## Load and setup the model
PATH = "/home/travail/Code/checkpoint/run_2023-03-24_11h28m47s/classic_noDataAugmentation_noDepth_resol100_seed0_subject5"
cfg = "./configs/train.yaml"
cfg = load_config(cfg)
cfg['num_classes'] = 29
model_type = 'Hands'

mod = model.ConvModel(cfg)
mod.load_state_dict(torch.load(PATH)['CONV'])
mod.to('cuda')
mod.eval()

global_acc = 0
total_nb_data = 0
for test_subject in range(1,6):
    ## Creating dataloader
    dataset_path = '/store/travail/data_sorted'

    dataset = cobotic_dataset.CHD(dataset_path, test=False, test_subject=test_subject)
    dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=False, drop_last=True)

    total_nb_data += len(dataloader)

    global_acc += test_model_on_data(mod, dataloader, model_type)

global_acc = (global_acc/total_nb_data) * 17/13
print()
print(f"La précision du modèle entraîné sur HANDS et testé sur Cobotics est de {global_acc}")

print("[INFO] Test of cobotic model on Hands data")

## Load and setup the model
PATH = "/store/travail/checkpoint/run_2023-10-30_22h50m54s/classic_cobotic_10s_v2_seed0_subject5"
cfg['num_classes'] = 10
model_type = 'Cobotic'

mod = model.ConvModel(cfg)
mod.load_state_dict(torch.load(PATH)['CONV'])
mod.to('cuda')
mod.eval()

global_acc = 0
total_nb_data = 0
for test_subject in range (1, 6):
    ## Creating the dataloader
    dataset = hand_gesture_dataset.HGD('/home/travail/Dataset/ndrczc35bt-1',
                                    depth=True,
                                    test=False,
                                    transform=False,
                                    data_aug_parameter={'noise_factor': 0.2, 'nb_black_boxes': 10, 'rotation_max': 30, 'black_boxes_size': 8, 'resol': 1},
                                    test_subject=test_subject)

    dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=True, drop_last=True)

    total_nb_data += len(dataloader)

    global_acc += test_model_on_data(mod, dataloader, model_type)

global_acc = (global_acc/total_nb_data) * 29/13
print()
print(f"La précision du modèle entraîné sur Cobotics et testé sur HANDS est de {global_acc}")

# [INFO] Test of Hands model on cobotic data

# La précision du modèle entraîné sur HANDS et testé sur Cobotics est de 0.3507365066108787
# [INFO] Test of cobotic model on Hands data

# La précision du modèle entraîné sur Cobotics et testé sur HANDS est de 0.9340512820512779