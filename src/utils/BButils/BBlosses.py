from torchvision import ops
import torch
import numpy as np

def calculate_IoU(target_BB, pred_BB, resol=1):
    """
    Both target_BB and pred_BB are like x1, y1, x2, y2
    """
    # box_iou requires tensors in entry
    if resol != 1:
        pred_BB = [pred / resol for pred in pred_BB]

    target_BB = torch.tensor(np.array([target_BB]))
    pred_BB = torch.tensor(np.array([pred_BB]))

    IoU = ops.box_iou(target_BB, pred_BB)
    return IoU.numpy()[0][0]