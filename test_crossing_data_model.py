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

# ## Load and setup the model
# PATH = [f"/home/travail/Code/checkpoint/run_2023-03-15_14h57m23s/classic_noDataAugmentation_noDepth_resol100_seed0_subject1",
#         f"/home/travail/Code/checkpoint/run_2023-03-15_12h50m54s/classic_noDataAugmentation_noDepth_resol100_seed0_subject2",
#         f"/home/travail/Code/checkpoint/run_2023-03-14_22h29m30s/classic_noDataAugmentation_noDepth_resol100_seed0_subject3",
#         f"/home/travail/Code/checkpoint/run_2023-03-21_16h14m37s/classic_noDataAugmentation_noDepth_resol100_seed0_subject4",
#         f"/home/travail/Code/checkpoint/run_2023-03-24_11h28m47s/classic_noDataAugmentation_noDepth_resol100_seed0_subject5"]
cfg = "./configs/train.yaml"
cfg = load_config(cfg)
# cfg['num_classes'] = 29
# model_type = 'Hands'
# global_acc_list = []

# for idx, path in enumerate(PATH):
#     mod = model.ConvModel(cfg)
#     mod.load_state_dict(torch.load(path)['CONV'])
#     mod.to('cuda')
#     mod.eval()

#     global_acc = 0
#     total_nb_data = 0
#     for test_subject in range(1,6):
#         ## Creating dataloader
#         dataset_path = '/store/travail/data_sorted'

#         dataset = cobotic_dataset.CHD(dataset_path, test=True, test_subject=test_subject)
#         dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=False, drop_last=True)

#         total_nb_data += len(dataloader)

#         global_acc += test_model_on_data(mod, dataloader, model_type)
#         print(f"For model {idx}, Accuracy for subject {test_subject}: DONE")

#     global_acc = (global_acc/total_nb_data) * 17/13
#     print()
#     print(f"La précision du modèle entraîné sur HANDS et testé sur Cobotics est de {global_acc}")
#     global_acc_list.append(global_acc)

# print(global_acc_list)

print("[INFO] Test of cobotic model on Hands data")

## Load and setup the model
PATH = [#f"/store/travail/checkpoint/run_2024-03-21_15h45m40s/classic_cobotic_20s_v2_seed0_subject1",
        f"/store/travail/checkpoint/run_2024-03-21_20h23m43s/classic_cobotic_20s_v2_seed0_subject2",
        #f"/store/travail/checkpoint/run_2024-03-18_16h43m03s/classic_cobotic_20s_v2_seed0_subject3",
        #f"/store/travail/checkpoint/run_2024-03-20_15h55m27s/classic_cobotic_20s_v2_seed0_subject4",
        #f"/store/travail/checkpoint/run_2024-03-20_22h36m55s/classic_cobotic_20s_v2_seed0_subject5"
        ]
cfg['num_classes'] = 10
model_type = 'Cobotic'
global_acc_list = []

for idx, path in enumerate(PATH):
    mod = model.ConvModel(cfg)
    mod.load_state_dict(torch.load(path)['CONV'])
    mod.to('cuda')
    mod.eval()

    global_acc = 0
    total_nb_data = 0
    for test_subject in range (1, 6):
        ## Creating the dataloader
        dataset = hand_gesture_dataset.HGD('/home/travail/Dataset/ndrczc35bt-1',
                                        depth=True,
                                        test=True,
                                        transform=False,
                                        data_aug_parameter={'noise_factor': 0.2, 'nb_black_boxes': 10, 'rotation_max': 30, 'black_boxes_size': 8, 'resol': 1},
                                        test_subject=test_subject)

        dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=True, drop_last=True)

        total_nb_data += len(dataloader)

        global_acc += test_model_on_data(mod, dataloader, model_type)
        print(f"For model {idx}, Accuracy for subject {test_subject}: DONE")

    global_acc = (global_acc/total_nb_data) * 29/13
    print()
    print(f"La précision du modèle entraîné sur Cobotics et testé sur HANDS est de {global_acc}")

    global_acc_list.append(global_acc)

print(global_acc_list)

# HANDS on COBOTIC dataset
# Model 0: 0.33435592689471877
# Model 1: 0.3290745020358757
# Model 2: 0.37535532586704573
# Model 3: 0.33599647848574893
# Model 4: 0.376794406210055

# COBOTIC on HANDS dataset
# Model 0: 0.9699487179487192
# Model 1: 0
# Model 2: 0.9580512820512825
# Model 3: 0.960512820512821
# Model 4: 0.9735384615384622
