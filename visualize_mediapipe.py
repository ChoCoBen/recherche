import cv2
import mediapipe as mp
from os.path import join
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src import hand_BB_dataset

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# '160_color.png' -> un doigt coupé

dataset_path = '/home/travail/Dataset/ndrczc35bt-1/Subject1/Color'
img_path = join(dataset_path, '160_color.png') #'/store/travail/image_for_test/pre_mediapipe.jpg'

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
        drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1)
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS, drawing_spec)
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
            # cv2.rectangle(annotated_image, (bounding_boxe[0], bounding_boxe[1]), (bounding_boxe[2], bounding_boxe[3]), (0, 0, 255), 3)

            # afficher l'image à l'intérieur de la bounding box
            w_r = bounding_boxe[0] - bounding_boxe[2]
            h_r = bounding_boxe[1] - bounding_boxe[3]
            image_in_BB = img[bounding_boxe[3] - int(0.2 * h_r):bounding_boxe[1]+ int(0.2 * h_r), bounding_boxe[2]- int(0.2 * w_r):bounding_boxe[0]+ int(0.2 * w_r)]
            plt.imshow(image_in_BB)
            plt.show()
        
        plt.imshow(annotated_image)
        plt.show()

colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 255, 0)]
print(bounding_boxes)
# Dessiner les rectangles sur l'image
for bounding_box in bounding_boxes:
    for j, coef in enumerate([0.2]):
        w_r = bounding_box[0] - bounding_box[2]
        h_r = bounding_box[1] - bounding_box[3]
        color = colors[j]
        cv2.rectangle(img, (bounding_box[0] + int(coef * w_r), bounding_box[1] + int(coef * h_r)), (bounding_box[2] - int(coef * w_r), bounding_box[3] - int(coef * h_r)), color, 3)
        #cv2.putText(img, f'{coef}', (bounding_box[0]+120, bounding_box[1] + 60 * (j + 1)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

# Afficher l'image
plt.imshow(img)
plt.show()