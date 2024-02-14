import numpy as np
import json

def associate_images(filtered_cams, deltaT):
    tmp_cams = [filtered_cam[:, 6].astype(float) for filtered_cam in filtered_cams]
    associated_images_groups = []

    images_already_added = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}}

    for i, timestamp_cam1 in enumerate(tmp_cams[0]):
        associated_group = [filtered_cams[0][i,0]]
        associated_group_num = [i]

        for cam in range(1, 6):
            for j, timestamp in enumerate(tmp_cams[cam]):
                if images_already_added[cam+1].get(j) is None and abs(timestamp - timestamp_cam1) <= deltaT:
                    associated_group.append(filtered_cams[cam][j,0])
                    associated_group_num.append(j)
                    break
        
        if len(associated_group) == 6:
            associated_images_groups.append(associated_group)

            for cam, image_num in enumerate(associated_group_num, start=1):
                images_already_added[cam][image_num] = True
    
    return associated_images_groups

def filter_array_by_user_gesture_zone(tab, user_gest_zone):
    index_lines_to_keep = np.array([user_gest_zone in chaine for chaine in tab[:, 0]])
    filtered_array = tab[index_lines_to_keep]
    return filtered_array

def calculate_pairing(min_delta_T):
    csv_cam1 = '/store/travail/data/txt1.csv'
    tab_cam1 = np.loadtxt(csv_cam1, delimiter=',', skiprows=1, dtype=str)

    csv_cam2 = '/store/travail/data/txt2.csv'
    tab_cam2 = np.loadtxt(csv_cam2, delimiter=',', skiprows=1, dtype=str)

    csv_cam3 = '/store/travail/data/txt3.csv'
    tab_cam3 = np.loadtxt(csv_cam3, delimiter=',', skiprows=1, dtype=str)

    csv_cam4 = '/store/travail/data/txt4.csv'
    tab_cam4 = np.loadtxt(csv_cam4, delimiter=',', skiprows=1, dtype=str)

    csv_cam5 = '/store/travail/data/txt5.csv'
    tab_cam5 = np.loadtxt(csv_cam5, delimiter=',', skiprows=1, dtype=str)

    csv_cam6 = '/store/travail/data/txt6.csv'
    tab_cam6 = np.loadtxt(csv_cam6, delimiter=',', skiprows=1, dtype=str)

    associated_group_per_user = {f'u{i}': None for i in range(1, 21)}
    tab_cameras = [tab_cam1, tab_cam2, tab_cam3, tab_cam4, tab_cam5, tab_cam6]

    for ui,u_value in associated_group_per_user.items():
        associated_group_per_value = {f'g{i}': None for i in range(1, 18)}
        for gi,g_value in associated_group_per_value.items():
            filtered_cams = []
            for ci, tab in zip([f'c{i}' for i in range(1, 7)], tab_cameras):
                user_gest_zone = '/' + ui + '/' + ci + '/' + gi + '_'
                filtered_cams.append(filter_array_by_user_gesture_zone(tab, user_gest_zone))
            delta_T = min_delta_T
            nb_groups_for_user_gesture = 0
            while nb_groups_for_user_gesture < 110:
                delta_T += 20
                associated_group_per_value[gi] = associate_images(filtered_cams, delta_T)
                nb_groups_for_user_gesture = len(associated_group_per_value[gi])


        associated_group_per_user[ui] = associated_group_per_value

    with open('association.json', 'w') as file:
        json.dump(associated_group_per_user, file)

def calculate_pairing_for_one_gesture(ui, gi, min_image):

    csv_cam1 = '/store/travail/data/txt1.csv'
    tab_cam1 = np.loadtxt(csv_cam1, delimiter=',', skiprows=1, dtype=str)

    csv_cam2 = '/store/travail/data/txt2.csv'
    tab_cam2 = np.loadtxt(csv_cam2, delimiter=',', skiprows=1, dtype=str)

    csv_cam3 = '/store/travail/data/txt3.csv'
    tab_cam3 = np.loadtxt(csv_cam3, delimiter=',', skiprows=1, dtype=str)

    csv_cam4 = '/store/travail/data/txt4.csv'
    tab_cam4 = np.loadtxt(csv_cam4, delimiter=',', skiprows=1, dtype=str)

    csv_cam5 = '/store/travail/data/txt5.csv'
    tab_cam5 = np.loadtxt(csv_cam5, delimiter=',', skiprows=1, dtype=str)

    csv_cam6 = '/store/travail/data/txt6.csv'
    tab_cam6 = np.loadtxt(csv_cam6, delimiter=',', skiprows=1, dtype=str)

    tab_cameras = [tab_cam1, tab_cam2, tab_cam3, tab_cam4, tab_cam5, tab_cam6]
    filtered_cams = []

    for ci, tab in zip([f'c{i}' for i in range(1, 7)], tab_cameras):
        user_gest_zone = '/' + ui + '/' + ci + '/' + gi + '_'
        filtered_cams.append(filter_array_by_user_gesture_zone(tab, user_gest_zone))
    delta_T = 60
    nb_groups_for_user_gesture = 0
    while nb_groups_for_user_gesture < min_image or delta_T > 800:
        delta_T += 20
        associated_group = associate_images(filtered_cams, delta_T)
        nb_groups_for_user_gesture = len(associated_group)
    
    for i, group in enumerate(associated_group):
        for j, image_path in enumerate(group):
            associated_group[i][j] = image_path.replace('/media/stage/T7/data/color','')
    
    with open('association.json', 'r') as file:
        association = json.load(file)

    association[ui][gi] = associated_group

    with open('association.json','w') as file:
        json.dump(association, file)

def calculate_stat_on_association():
    min = 1000
    max = 0
    mean = 0
    with open('association.json', "r") as file:
        associations = json.load(file)
        for ui, u_value in associations.items():
            for gi, g_value in u_value.items():
                mean += len(g_value)/(17*20)
                if len(g_value) < min:
                    min = len(g_value)
                    ui_min = ui
                    gi_min = gi

                elif len(g_value) > max:
                    max = len(g_value)
                    ui_max = ui
                    gi_max = gi
    
    print(f'min:{min}, ui_min:{ui_min}, gi_min: {gi_min}')
    print(f'max:{max}, ui_max:{ui_max}, gi_max: {gi_max}')
    print(f'mean:{mean}')

delta_T = 20

# calculate_pairing(delta_T)
# calculate_stat_on_association()