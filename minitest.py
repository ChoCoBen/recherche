import torch

tens = torch.tensor([[5, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

tens = (tens-torch.min(tens))/(torch.max(tens)-torch.min(tens))

print(tens)