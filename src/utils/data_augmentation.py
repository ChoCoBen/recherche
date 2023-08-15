import torch
import torchvision.transforms.functional as TF

def gaussian_noise(image, noise_factor, width, height):
    x1, x2, y1, y2 = calculate_limit_images(image, width, height)
    noisy = image.clone()
    noise = torch.randn_like(noisy[:, x1:x2, y1:y2])
    noisy[:, x1:x2, y1:y2] += noise * torch.rand(1).item() * noise_factor
    noisy = torch.clip(noisy,0.,1.)
    return noisy

def random_blocks(image, nb_black_boxes, black_boxes_size):
    image = image.clone()
    h, w = black_boxes_size, black_boxes_size
    img_size = image.shape[1]
    for k in range(torch.randint(5, nb_black_boxes+1, (1,)).item()): # Nb of black boxes randomly choose between 5 and the number max chosen
        y, x = torch.randint(0, img_size - w, (2,))
        image[:, y:y+h, x:x+w] = 0
    return image

def rotation(image, rotation_max):
    angle = torch.rand(1) * 2 * rotation_max - rotation_max
    angle = angle.item()
    image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
    return image

def calculate_limit_images(image, width, height):
    img_size = image.shape[1]
    print(f'img_size:{img_size}, height:{height}, width:{width}')

    minxy, maxxy = min(height, width), max(height, width)
    coef = (img_size / maxxy)

    if maxxy == height:
        real_width = int(coef * width)
        diff = img_size - real_width
        return int(diff / 2), img_size - int(diff / 2), 0, img_size
    
    else:
        real_height = int(coef * height)
        diff = img_size - real_height
        return 0, img_size, int(diff / 2), img_size - int(diff / 2)

