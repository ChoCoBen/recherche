import mediapipe as mp
from torch.utils.data import DataLoader
import numpy as np
from src import hand_BB_dataset
from src.utils.BButils import BBhelper, BBlosses
from src.utils.helper import *
from src.core import model
import cv2
import matplotlib.pyplot as plt

def create_dataloader(test_subject):
    test_dataset = hand_BB_dataset.HBBD('/home/travail/Dataset/ndrczc35bt-1',
                                    test=True, 
                                    test_subject=test_subject)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, drop_last=True)
    return test_dataloader

def calculate_mean_iou(test_subject, coef_BB):
    test_dataloader = create_dataloader(test_subject)

    hands_detected = 0
    IoUs = []
    for data in test_dataloader:
        image, first_hand_BB_target, second_hand_BB_target = data['image'], data['First_hand'][0].numpy(), data['Second_hand'][0].numpy()
        
        image_np = image[0].numpy()
        img_cv2 = image_np.astype(np.uint8)

        h, w, c = image_np.shape


        mp_hands = mp.solutions.hands
        with mp_hands.Hands() as hands:
            results = hands.process(img_cv2)
            bounding_boxes = BBhelper.calculate_mediapipe_BB(results, w, h, coef_BB)

            if len(bounding_boxes) == 0:
                IoU = [0, 0]
            elif len(bounding_boxes) == 1:
                hands_detected += 1
                IoU_with_first_hand = BBlosses.calculate_IoU(first_hand_BB_target, bounding_boxes[0])
                IoU_with_second_hand = BBlosses.calculate_IoU(second_hand_BB_target, bounding_boxes[0])

                IoU = [max(IoU_with_first_hand, IoU_with_second_hand), 0]
            else:
                hands_detected += 2
                IoU_FT_FP = BBlosses.calculate_IoU(first_hand_BB_target, bounding_boxes[0])
                IoU_ST_SP = BBlosses.calculate_IoU(second_hand_BB_target, bounding_boxes[1])

                IoU_ST_FP = BBlosses.calculate_IoU(second_hand_BB_target, bounding_boxes[0])
                IoU_FT_SP = BBlosses.calculate_IoU(first_hand_BB_target, bounding_boxes[1])

                IoU = [IoU_FT_FP, IoU_ST_SP] if IoU_FT_FP + IoU_ST_SP > IoU_ST_FP + IoU_FT_SP else [IoU_ST_FP, IoU_FT_SP]
            IoUs.append(np.mean(IoU))


    percentage = hands_detected / len(IoUs) / 2 * 100
    IoU_percentage = np.mean(IoUs)

    print(f"Pour le sujet n°{test_subject} avec un coefficient de {coef_BB}, les résulats sont les suivants:")
    print("") # Espace  pour plus de lisibilité
    print(f'Le pourcentage de main détecté est de {percentage}%')
    print(f"La valeur de l'IoU moyen sur ce set de test est de {IoU_percentage}")
    print("") # Espace  pour plus de lisibilité

    return IoU_percentage

def search_best_coef():
    coefs = {0.05:None, 0.1:None, 0.15:None, 0.18:None, 0.19:None, 0.20:None, 0.21: None, 0.22:None, 0.25:None}
    test_subjects = [1, 2, 3, 4, 5]
    for coef in coefs.keys():
        print(f'coef = {coef}')
        iou_mean_subjects = []
        for test_subject in test_subjects:
            print(f'test_subject = {test_subject}')
            iou_mean_subjects.append(calculate_mean_iou(test_subject, coef))
        coefs[coef] = np.mean(iou_mean_subjects)
    
    print(coefs)

def load_model(model_path):
    PATH = model_path
    cfg = "./configs/train.yaml"
    cfg = load_config(cfg)

    mod = model.ConvModel(cfg)
    mod.load_state_dict(torch.load(PATH)['CONV'])
    mod.eval()
    return mod

def calculate_acc_after_mediapipeBB(test_subject, model_path, coef_BB=0.2):
    test_dataloader = create_dataloader(test_subject)
    mod = load_model(model_path)

    total_hands = len(test_dataloader) * 2
    hands_detected = 0
    number_hand_correctly_predicted = 0

    for data in test_dataloader:
        image = data['image'][0].numpy()
        first_hand_BB_target = data['First_hand'][0].numpy()
        second_hand_BB_target = data['Second_hand'][0].numpy()
        first_hand_ges_target = int(data['First_gesture'][0].numpy())
        second_hand_ges_target = int(data['Second_gesture'][0].numpy())
        
        img_cv2 = image.astype(np.uint8)

        h, w, c = image.shape

        mp_hands = mp.solutions.hands
        with mp_hands.Hands() as hands:
            results = hands.process(img_cv2)
            bounding_boxes = BBhelper.calculate_mediapipe_BB(results, w, h, coef_BB)

        hands_detected += len(bounding_boxes)
        for bounding_box in bounding_boxes:
            image_cutted = BBhelper.setup_image_for_model(img_cv2, bounding_box)
            prediction = mod(image_cutted.view(1, 4, 100, 100))
            prediction = torch.sigmoid(prediction)
            indice_pred = torch.argmax(prediction).item()
            if indice_pred == first_hand_ges_target or indice_pred == second_hand_ges_target:
                number_hand_correctly_predicted += 1
        
    print(f'Le nombre total de main est {total_hands}')
    print(f'Le nombre de main détecté par mediapipe est {hands_detected}')
    print(f'Sur les mains détectés, le nombre de main correctement prédis est {number_hand_correctly_predicted}')
    print(f'La précision totale est donc: {number_hand_correctly_predicted/total_hands}%')
    print(f'La précision seulement sur les mains détectés est: {number_hand_correctly_predicted/hands_detected}')

        
for test_subject in [1,2,3,4,5]:
    calculate_mean_iou(test_subject=test_subject, coef_BB=0.2)
#search_best_coef()

model_path = "../checkpoint/run_2023-03-15_14h57m23s/classic_noDataAugmentation_noDepth_resol100_seed0_subject1"
#calculate_acc_after_mediapipeBB(test_subject=1, model_path=model_path)
