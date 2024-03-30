import os
import json
import shutil

with open('/home/travail/Code/Static_Resnet/cobotic_dataset_creation/association_annotation.json', 'r') as f:
    associations = json.load(f)

path_COBOTIC = '/run/media/cohub/Elements/COBOTIC_DATASET'
path_data_sorted = '/store/travail/data_sorted'

for ui, u_value in associations.items():
    print(f'[INFO] Processing user {ui}')
    ui_path = os.path.join(path_COBOTIC, ui)
    if not os.path.exists(ui_path):
        os.makedirs(ui_path)
    for gi, g_value in u_value.items():
        print(f'[INFO] Processing gesture {gi}')
        gi_path = os.path.join(ui_path, gi)
        if not os.path.exists(gi_path):
            os.makedirs(gi_path)
        if not os.path.exists(os.path.join(gi_path, 'color')):
            os.makedirs(os.path.join(gi_path, 'color'))
        if not os.path.exists(os.path.join(gi_path, 'depth')):
            os.makedirs(os.path.join(gi_path, 'depth'))
        if not os.path.exists(os.path.join(gi_path, 'annotation')):
            os.makedirs(os.path.join(gi_path, 'annotation'))
        
        for idx, group in enumerate(g_value):
            for image_annotation in group:
                image = image_annotation['image']
                ci = image.split('/')[2]
                name_image = image.split('/')[3]
                annotation = image_annotation['annotation']

                image_path = os.path.join(path_data_sorted, ui, gi, ci, 'color', name_image)
                depth_path = os.path.join(path_data_sorted, ui, gi, ci, 'depth', name_image)
                annotation_path = os.path.join(path_data_sorted, ui, gi, ci, 'annotation', annotation) if annotation is not None else None
                
                new_name_image = f"{idx}_{ci}.png"
                new_annotation_image = f"{idx}_{ci}.txt"

                new_color_path = os.path.join(gi_path, 'color', new_name_image)
                new_depth_path = os.path.join(gi_path, 'depth', new_name_image)
                new_annotation_path = os.path.join(gi_path, 'annotation', new_annotation_image)

                # if not os.path.exists(image_path):
                #     print(image_path)
                # if not os.path.exists(depth_path):
                #     print(depth_path)
                # if annotation_path is not None and not os.path.exists(annotation_path):
                #     print(annotation_path)
                
                if not os.path.exists(new_color_path):
                    shutil.copy(image_path, new_color_path)
                if not os.path.exists(new_depth_path):
                    shutil.copy(depth_path, new_depth_path)
                if not os.path.exists(new_annotation_path):
                    if annotation_path is not None:
                        shutil.copy(annotation_path, new_annotation_path)
                    else:
                        with open(new_annotation_path, 'w') as f:
                            f.write('')


        