import cv2
import mediapipe as mp
from os.path import join
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src import hand_BB_dataset

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

dataset_path = '/home/travail/Dataset/ndrczc35bt-1/Subject1/Color'
img_path = join(dataset_path, '160_color.png')

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, c = img.shape

plt.imshow(img)
plt.show()

with mp_hands.Hands() as hands:
    result = hands.process(img)
    
    if not result.multi_hand_landmarks:
        print('[INFO] No hands detected')
    else:
        annotated_image = img.copy()
        bounding_boxes = []
        # For every hands detected
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            maxlmx, maxlmy = 0, 0
            minlmx, minlmy = 1, 1

            # For every key points for each hand
            for id, lm in enumerate(hand_landmarks.landmark):
                maxlmx = lm.x if lm.x > maxlmx else maxlmx
                minlmx = lm.x if lm.x < minlmx else minlmx
                maxlmy = lm.y if lm.y > maxlmy else maxlmy
                minlmy = lm.y if lm.y < minlmy else minlmy            
            
            bounding_boxe = [int(maxlmx * w), int(maxlmy * h), int(minlmx * w), int(minlmy * h)]
            bounding_boxes.append(bounding_boxe)
        
        plt.imshow(annotated_image)
        plt.show()


# Dessiner les rectangles sur l'image
for bounding_box in bounding_boxes:
    w_r = bounding_box[0] - bounding_box[2]
    h_r = bounding_box[1] - bounding_box[3]
    cv2.rectangle(img, (bounding_box[0] + int(0.1 * w_r), bounding_box[1] + int(0.1 * h_r)), (bounding_box[2] - int(0.1 * w_r), bounding_box[3] - int(0.1 * h_r)), (0, 0, 255), 2)

# Afficher l'image
plt.imshow(img)
plt.show()