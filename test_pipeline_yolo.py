import torch
from ultralytics import YOLO
from src.core import model
import json
import numpy as np
import cv2
from os.path import join
import os
import torch.nn as nn 
from src.utils.cobotic_helper import set_image
from src.utils.helper import load_config

# Load the model
cobotics_weights_detection = [
    f'/store/travail/Detection/COBOTIC_weights/yolov8m_COBOTIC_test{i}.pt' for i in range(1, 6)
]

cobotics_weights_recognition = [f"/store/travail/checkpoint/run_2024-03-21_15h45m40s/classic_cobotic_20s_v2_seed0_subject1",
        f"/store/travail/checkpoint/run_2024-03-27_09h09m02s/classic_cobotic_20s_v2_seed0_subject2",
        f"/store/travail/checkpoint/run_2024-03-18_16h43m03s/classic_cobotic_20s_v2_seed0_subject3",
        f"/store/travail/checkpoint/run_2024-03-20_15h55m27s/classic_cobotic_20s_v2_seed0_subject4",
        f"/store/travail/checkpoint/run_2024-03-20_22h36m55s/classic_cobotic_20s_v2_seed0_subject5"
        ]

def load_mod_cfg_data(test_subject):
    print(f"Loading model for subject {test_subject}")
    cfg = "./configs/train.yaml"
    cfg = load_config(cfg)

    ## Load and setup the model
    PATH = cobotics_weights_recognition[test_subject-1]
    cfg['num_classes'] = 10

    mod = model.ConvModel(cfg)
    mod.load_state_dict(torch.load(PATH)['CONV'])
    mod.to('cuda')
    mod.eval()

    detection_model = YOLO(cobotics_weights_detection[test_subject-1])

    path_to_data = "/store/travail/data_sorted"

    subjects = ['u'+str(i) for i in range (1, 11)]
    subject_to_subjects = {1:(subjects[2:10],subjects[0:2]), 2:(subjects[0:2]+subjects[4:10],subjects[2:4]), 3:(subjects[0:4]+subjects[6:10],subjects[4:6]), 4:(subjects[0:6]+subjects[8:10],subjects[6:8]), 5:(subjects[0:8],subjects[8:10])}
    train_subjects, test_subjects = subject_to_subjects[test_subject]

    return cfg, mod, test_subjects, path_to_data, detection_model

def calculate_pred_for_every_images(test_subject, use_depth=True):
    print(f"Calculating predictions for subject {test_subject}")
    cfg, recognition_model, test_subjects, path_to_data, detection_model = load_mod_cfg_data(test_subject)
    with open('cobotic_dataset_creation/association.json', 'r') as f:
        association = json.load(f)
    
    with torch.no_grad():
        for ui in test_subjects:
            u_value = association[ui]
            subject_path = join(path_to_data, ui)
            for gi, g_value in u_value.items():
                gi_path = join(subject_path, gi)
                for num_associated_images,associated_images in enumerate(g_value):
                    for i,image_file in enumerate(associated_images):
                        torch.cuda.empty_cache()
                        ci = 'c'+str(i+1)
                        ci_path = join(gi_path, ci)
                        if 'annotation' in os.listdir(ci_path):
                            image_file = image_file[7:] if int(ui[1:]) < 10 else image_file[8:]

                            color_path = join(path_to_data, ui, gi, ci, 'color', image_file)
                            depth_path = join(path_to_data, ui, gi, ci, 'depth', image_file)

                            detection_prediction = calculate_BB(detection_model, color_path)
                            torch.cuda.empty_cache()
                            if detection_prediction is not None:
                                image = np.array([color_path] + [depth_path] + detection_prediction)
                                image = setup_image(image, use_depth)

                                prediction = recognition_model(image.unsqueeze(0).to('cuda'))
                                torch.cuda.empty_cache()
                                prediction = nn.functional.softmax(prediction, dim=1).squeeze(0).cpu()
                                
                                
                                association[ui][gi][num_associated_images][i] = prediction
                            else:
                                association[ui][gi][num_associated_images][i] = None
    with open('/store/travail/pred_for_all_associated_complete_pipeline.json', 'r') as f:
        new_association = json.load(f)
    new_association[test_subjects[0]] = association[test_subjects[0]]
    new_association[test_subjects[1]] = association[test_subjects[1]]
    with open('/store/travail/pred_for_all_associated_complete_pipeline.json', 'w') as f:
        json.dump(new_association, f)

def setup_image(image, use_depth=True):
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

def calculate_BB(detection_model, color_path):
    results = detection_model(color_path, verbose=False)
    results = results[0].boxes.cpu()
    if len(results.data) == 0:
        return None
    
    indice_max = torch.argmax(results.data[:, 4])

    result = results.data[indice_max]
            
    result_prob = result[-2].item()
    if result_prob > 0.5:
        return results.xywhn[indice_max].squeeze().tolist()
    else: return None

def check_annotated(ui, image_file, ci_path):
    annotation_file = image_file.replace('.png', '.txt')[7:] if int(ui[1:]) < 10 else image_file.replace('.png', '.txt')[8:]
    return annotation_file in os.listdir(join(ci_path, 'annotation'))

for i in range(1,6):
    calculate_pred_for_every_images(i)
    

