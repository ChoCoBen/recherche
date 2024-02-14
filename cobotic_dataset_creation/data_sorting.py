import sys
sys.path.append('/home/travail/Code/Static_Resnet')

from images_pairing import *
import mediapipe as mp
import json
import cv2
from src.utils.BButils import BBhelper
import shutil
import os
from PIL import Image
from IPython.display import display

color_path = '/store/travail/data/color'

zone_to_camera = {1:[3,5], 2:[0,4,5], 3:[0,2,4], 4:[1,2]}

def after_p(chaine):
    for i in range(len(chaine) - 1):
        if chaine[i] == 'p':
            return chaine[i + 1]
    return None

def testing_annotation_with_mediapipe(zone_to_camera, color_path):
    with open('association.json', "r") as file:
        associations = json.load(file)

    nb_total = 0
    nb_total_detected = 0

    for ui, u_value in associations.items():
        if ui == 'u11':
            break
        
        for gi, g_value in u_value.items():
            nb_image_group_with_hand_detected = 0
            nb_image_for_one_gesture = len(g_value)
            camera = zone_to_camera[int(after_p(g_value[0][0]))]

            for image_group in g_value:
                for cam in camera:       
                    image_path = color_path + image_group[cam]
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    h, w, c = image.shape

                    mp_hands = mp.solutions.hands
                    with mp_hands.Hands() as hands:
                        results = hands.process(image)
                        bounding_boxes = BBhelper.calculate_mediapipe_BB(results, w, h, 0.2)
                    
                    if len(bounding_boxes) != 0:
                        nb_image_group_with_hand_detected += 1
            nb_total_detected += nb_image_group_with_hand_detected
            nb_total += nb_image_for_one_gesture
            
            print(f"Pour le user {ui} et le geste {gi} pour les caméras {[cam+1 for cam in camera]}, il y a {nb_image_group_with_hand_detected} groupes d'images où une main a été détecté sur {nb_image_for_one_gesture}.")
    print(f"Pour l'ensemble des données, il y a {nb_total_detected} groupes d'images où une main a été détecté sur {nb_total}.")

def moving_associated_data(folder):
    with open('association.json', "r") as file:
        associations = json.load(file)
    
    if (os.path.exists(folder)):
        print(f'[WARNING] Le dossier {folder} existe déjà')
        return 0
    else: os.mkdir(folder)
    
    for ui, u_value in associations.items():
        os.mkdir(folder+'/'+ui)
        for gi, g_value in u_value.items():
            os.mkdir(folder+'/'+ui+'/'+gi)
            for i,ci in enumerate(['c'+str(i) for i in range(1,7)]):
                os.mkdir(folder+'/'+ui+'/'+gi+'/'+ci)
                os.mkdir(folder+'/'+ui+'/'+gi+'/'+ci+'/color')
                os.mkdir(folder+'/'+ui+'/'+gi+'/'+ci+'/depth')
                for group in g_value:
                    color_source = '/store/travail/data/color'
                    depth_source = '/store/travail/data/depth'
                    shutil.copy(color_source+group[i],folder+'/'+ui+'/'+gi+'/'+ci+'/color')
                    shutil.copy(depth_source+group[i],folder+'/'+ui+'/'+gi+'/'+ci+'/depth')

def delete_folder(folder):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

def moving_associated_data_for_one_gesture(ui, gi, folder):
    with open('association.json', "r") as file:
        associations = json.load(file)

    for i,ci in enumerate(['c'+str(i) for i in range(1,7)]):
        color_folder = folder+'/'+ui+'/'+gi+'/'+ci+'/color'
        depth_folder = folder+'/'+ui+'/'+gi+'/'+ci+'/depth'
        delete_folder(color_folder)
        delete_folder(depth_folder)
        for group in associations[ui][gi]:
            color_source = '/store/travail/data/color'
            depth_source = '/store/travail/data/depth'
            shutil.copy(color_source+group[i], color_folder)
            shutil.copy(depth_source+group[i], depth_folder)

def select_manually_bad_groups():
    with open('association.json', "r") as file:
        associations = json.load(file)
    
    with open('manually_selected.json', "r") as file:
        manual_selection = json.load(file)    
    
    for ui, u_value in associations.items():
        try : manual_selection[ui]
        except KeyError: manual_selection[ui] = {}
        for gi, g_value in u_value.items():
            try:
                manual_selection[ui][gi]
            except KeyError:
                for group in g_value:
                    camera = zone_to_camera[int(after_p(group[0]))]
                    for image_path in [group[cam] for cam in camera]:
                        color_path = '/store/travail/data/color'
                        path = color_path + image_path

                        img = Image.open(path)
                        display(img)
                    
                    choice = input("Voulez-vous conserver ce groupe d'images ? (Y/n) ").strip().lower()
                    while choice not in ["y", "n"]:
                        print("Réponse invalide. Veuillez répondre 'Y' ou 'N'.")
                        choice = input("Voulez-vous conserver ce groupe d'images ? (Y/n) ").strip().lower()
                    if choice == 'y':
                        manual_selection[ui][gi] = {'group':group, 'correct_group' : True}
                    elif choice == "n":
                        manual_selection[ui][gi] = {'group':group, 'correct_group' : False}
                with open('manually_selected.json', "w") as file:
                    json.dump(manual_selection, file)  
                print(f'You are done with {gi} for {ui}. File is saved')
        print(f'You are done with {ui}')

def count_associate_and_correct_for_ui_gi(associations, ui, gi):    
    associated_groups = associations[ui][gi]
    cameras = zone_to_camera[int(after_p(associated_groups[0][0]))]
    
    cis = ['c'+str(i+1) for i in cameras]
    cis_path = [('/store/travail/data_sorted/'+ui+'/'+gi+'/'+ci+'/annotation', int(ci[1])-1) for ci in cis]

    count = 0
    for group in associated_groups:
        at_least_one_hand_detected_in_group = False
        for ci_path in cis_path:
            print(group[ci_path[1]][7:])
            at_least_one_hand_detected_in_group = at_least_one_hand_detected_in_group or group[ci_path[1]][7:].replace('png', 'txt') in os.listdir(ci_path[0])
            
        if at_least_one_hand_detected_in_group:
            count+=1
    
    if count<100: print(f"Il n'y a pas suffisamment de mains détectés pour le geste {gi} de l'utilisateur {ui}: count = {count}")

def count_associate_and_correct_for_all():
    with open('association.json', "r") as file:
        associations = json.load(file)
    
    for ui in ['u'+str(i) for i in range(1,11)]:
        for gi in ['g'+str(i) for i in range(1,18)]:
            count_associate_and_correct_for_ui_gi(associations, ui, gi)

def verify_annotation(folder):
    if os.path.exists(folder + '/annotation'):
        annotation_files = os.listdir(folder + '/annotation')
    
        image_files = os.listdir(folder + '/color')

        for annotation_file in annotation_files:
            if (not annotation_file.replace('txt', 'png') in image_files) and annotation_file != 'classes.txt':
                your_input = input(f'{annotation_file} for {folder}?') or False
                if not your_input:
                    os.remove(folder + '/annotation' + '/' + annotation_file)
            
            elif os.path.getsize(folder + '/annotation' + '/' + annotation_file) == 0:
                os.remove(folder + '/annotation' + '/' + annotation_file)
                print(f'{annotation_file} deleted')

                

def increase_associate_for_one_gesture_one_user(ui, gi, min_image):
    calculate_pairing_for_one_gesture(ui, gi, min_image)
    moving_associated_data_for_one_gesture(ui, gi, '/store/travail/data_sorted')
    for ci in ['c' + str(i) for i in range(1,7)]:
        verify_annotation('/store/travail/data_sorted/' + ui +'/'+ gi +'/'+ci)

def verify_all_annotation():
    for ui in ['u'+str(i) for i in range (1, 11)]:
        for gi in ['g'+str(i) for i in range (1, 18)]:
            for ci in ['c' + str(i) for i in range(1,7)]:
                verify_annotation('/store/travail/data_sorted/' + ui +'/'+ gi +'/'+ci)
        print(f'ALL GOOD FOR {ui}')


folder2 = '/store/travail/data_sorted'
# with open('association.json', "r") as file:
#         associations = json.load(file)

count_associate_and_correct_for_all()

# moving_associated_data(folder2)
# increase_associate_for_one_gesture_one_user('u6', 'g16', 115)
# verify_all_annotation()
# verify_annotation('/store/travail/data_sorted/u1/g3/c1')

#testing_annotation_with_mediapipe(zone_to_camera, color_path)
#Pour l'ensemble des données, il y a 3846 groupes d'images où une main a été détecté sur 19733.