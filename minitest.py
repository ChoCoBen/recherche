import cv2
import matplotlib.pyplot as plt

PATH = '/store/travail/data_sorted/u7/g7/c4/depth/g7_p1_15.png'
# Read the downloaded image
image = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)

# Apply a color map to the image
color_image = cv2.applyColorMap(image, cv2.COLORMAP_JET)

# Display the color image
plt.imshow(color_image)
plt.axis('off')
plt.show()
