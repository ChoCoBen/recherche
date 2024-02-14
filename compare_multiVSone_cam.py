import torch
from src.core import model
from src.utils.helper import *
from os.path import join
import numpy as np
import cv2
from src.utils.cobotic_helper import set_image
import json
import torch.nn as nn

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
    model_type = 'Cobotic'

    mod = model.ConvModel(cfg)
    mod.load_state_dict(torch.load(PATH)['CONV'])
    mod.to('cuda')
    mod.eval()

    path_to_data = "/store/travail/data_sorted"

    subjects = ['u'+str(i) for i in range (1, 11)]
    subject_to_subjects = {1:(subjects[2:10],subjects[0:2]), 2:(subjects[0:2]+subjects[4:10],subjects[2:4]), 3:(subjects[0:4]+subjects[6:10],subjects[4:6]), 4:(subjects[0:6]+subjects[8:10],subjects[6:8]), 5:(subjects[0:8],subjects[8:10])}
    train_subjects, test_subjects = subject_to_subjects[test_subject]

    return cfg, mod, test_subjects, path_to_data

def compare_multi_vs_one_for_one_subject(test_subject, use_depth=False):
    cfg, mod, test_subjects, path_to_data = load_mod_cfg_data(test_subject)
    with open('cobotic_dataset_creation/association.json', 'r') as f:
        association = json.load(f)
    
    gi_to_i = {'g1':1, 'g2':2, 'g3':3, 'g4':4, 'g5':5, 'g6':6, 'g7':7, 'g8':8, 'g9':9, 'g10':1, 'g11':2, 'g12':3, 'g13':4, 'g14':5, 'g15':6, 'g16':10, 'g17':10}

    ui_to_gi = {}

    nb_total, nb_total_correct = 0, 0

    for ui in test_subjects:
        u_value = association[ui]
        subject_path = join(path_to_data, ui)
        for gi, g_value in u_value.items():
            gi_to_associated_images = []
            gi_path = join(subject_path, gi)
            for num_associated_images,associated_images in enumerate(g_value):
                total_prediction = torch.zeros(10)
                predictions_for_group = []
                for i,image_file in enumerate(associated_images):
                    prediction = torch.zeros(10)
                    ci_path = join(gi_path, 'c'+str(i+1))
                    if 'annotation' in os.listdir(ci_path):
                        annotation_file = image_file.replace('.png', '.txt')[7:] if int(ui[1:]) < 10 else image_file.replace('.png', '.txt')[8:]
                        if annotation_file in os.listdir(join(ci_path, 'annotation')):
                            annotation_path = join(join(ci_path, 'annotation'), annotation_file)

                            with open(annotation_path, 'r') as file:
                                bounding_box = file.readline().strip().split()
                                bounding_box = [float(nombre) for nombre in bounding_box[-4:]]

                                color_path = join(join(ci_path, 'color'), image_file)
                                depth_path = join(join(ci_path, 'depth'), image_file)

                                image = np.array([ci_path+'/color/'+color_path[7:]] + [ci_path+'/depth/'+depth_path[7:]] + bounding_box)
                                image = setup_image(image, use_depth)

                                prediction = mod(image.unsqueeze(0).to('cuda'))
                                prediction = nn.functional.softmax(prediction, dim=1).squeeze(0).cpu()
                                total_prediction += prediction
                                predictions_for_group.append(prediction.tolist())
                        association[ui][gi][num_associated_images] = predictions_for_group
                gesture_predicted = torch.argmax(total_prediction).item() + 1
                correctly_predicted = gi_to_i[gi] == gesture_predicted
                nb_total += 1
                nb_total_correct += correctly_predicted

                gi_to_associated_images.append(correctly_predicted)
        ui_to_gi[ui] = gi_to_associated_images
    with open('/store/travail/pred_for_all_associated.json', 'r') as f:
        new_association = json.load(f)
    new_association[test_subjects[0]] = association[test_subjects[0]]
    new_association[test_subjects[1]] = association[test_subjects[1]]
    with open('/store/travail/pred_for_all_associated.json', 'w') as f:
        json.dump(new_association, f)
    print(f'La précision quand on prend des groupes est de {nb_total_correct/nb_total} pour le set de test {test_subject}.')

def setup_image(image, use_depth=False):
        rgb = cv2.imread(image[0])
        depth = cv2.imread(image[1], cv2.IMREAD_UNCHANGED)

        total_height, total_width = rgb.shape[:2]

        BB = image[2:]
        
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
        return image

for i in range(1,6):
    compare_multi_vs_one_for_one_subject(i)
                        
# La précision quand on prend des groupes est de 0.9165460684997588 pour le set de test 1.
# La précision quand on prend des groupes est de 0.9260590500641849 pour le set de test 2.
# La précision quand on prend des groupes est de 0.8866133866133866 pour le set de test 3.
# La précision quand on prend des groupes est de 0.8958875585632483 pour le set de test 4.
# La précision quand on prend des groupes est de 0.9240769630785232 pour le set de test 5.