import json
import os
import random

with open('/home/travail/Code/Static_Resnet/cobotic_dataset_creation/association.json', 'r') as f:
    associations = json.load(f)

ci_list = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']
association_final = {}

for ui, u_value in associations.items():
    new_ui_value = {}

    for gi, g_value in u_value.items():
        path_gi = f"/store/travail/data_sorted/{ui}/{gi}"
        ci_for_ui_gi = []

        for idx, ci in enumerate(ci_list):
            path_ci = os.path.join(path_gi, ci)
            if os.path.exists(os.path.join(path_ci, 'annotation')):
                ci_for_ui_gi.append([idx, ci])

        new_associated_images = []
        for associated_images in g_value:
            is_group_annoted = False
            new_group = []
            for idx, ci in ci_for_ui_gi:
                path_ci = os.path.join(path_gi, ci)
                path_annotation = os.path.join(path_ci, 'annotation')
                image_file = associated_images[idx]
                annotation_file = image_file.replace('.png', '.txt')[7:] if int(ui[1:]) < 10 else image_file.replace('.png', '.txt')[8:]
                if annotation_file in os.listdir(path_annotation):
                    is_group_annoted = True
                    new_group.append({'image': image_file, 'annotation': annotation_file})
                else: new_group.append({'image': image_file, 'annotation': None})
            
            if is_group_annoted:
                new_associated_images.append(new_group)
        
        new_associated_images = random.sample(new_associated_images, 100)
        new_ui_value[gi] = new_associated_images
    association_final[ui] = new_ui_value

with open('/home/travail/Code/Static_Resnet/cobotic_dataset_creation/association_annotation.json', 'w') as f:
    json.dump(association_final, f)