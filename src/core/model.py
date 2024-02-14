import torch.nn as nn
from torch.optim import Adam
import torch
from src.utils.helper import *
from src.utils.losses import *

import wandb

class ConvModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg['num_classes']
        self.result_history = []

        ## Create the different units of the network

        ## 1st conv unit
        self.conv11 = nn.Conv2d(4, 50, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv12 = nn.Sequential(
            nn.Conv2d(50, 50, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.Conv2d(50, 50, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.Conv2d(50, 50, (3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.bn1 = nn.BatchNorm2d(50)
        self.maxpool1 = nn.MaxPool2d((2, 2))

        ## 2nd conv unit
        self.conv21 = nn.Conv2d(50, 125, (3, 3), stride=(1, 1), padding=(0, 0))
        self.conv22 = nn.Sequential(
            nn.Conv2d(125, 125, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(125),
            nn.ReLU(),
            nn.Conv2d(125, 125, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(125),
            nn.ReLU(),
            nn.Conv2d(125, 125, (3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.bn2 = nn.BatchNorm2d(125)
        self.maxpool2 = nn.MaxPool2d((2, 2))

        ## 3rd conv unit
        self.conv31 = nn.Conv2d(125, 250, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv32 = nn.Sequential(
            nn.Conv2d(250, 250, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(250),
            nn.ReLU(),
            nn.Conv2d(250, 250, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(250),
            nn.ReLU(),
            nn.Conv2d(250, 250, (3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.bn3 = nn.BatchNorm2d(250)
        self.maxpool3 = nn.MaxPool2d((2, 2))
        
        ## 4th conv unit
        self.conv41 = nn.Conv2d(250, 420, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv42 = nn.Sequential(
            nn.Conv2d(420, 420, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(420),
            nn.ReLU(),
            nn.Conv2d(420, 420, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(420),
            nn.ReLU(),
            nn.Conv2d(420, 420, (3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.bn4 = nn.BatchNorm2d(420)
        self.maxpool4 = nn.MaxPool2d((2, 2))

        ## 5th conv unit
        self.conv51 = nn.Conv2d(420, 540, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv52 = nn.Sequential(
            nn.Conv2d(540, 540, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(540),
            nn.ReLU(),
            nn.Conv2d(540, 540, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(540),
            nn.ReLU(),
            nn.Conv2d(540, 540, (3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.bn5 = nn.BatchNorm2d(540)
        self.maxpool5 = nn.MaxPool2d((2, 2))

        ## 6th conv unit
        self.conv61 = nn.Conv2d(540, 720, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv62 = nn.Sequential(
            nn.Conv2d(720, 720, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(720),
            nn.ReLU(),
            nn.Conv2d(720, 720, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(720),
            nn.ReLU(),
            nn.Conv2d(720, 720, (3, 3), stride=(1, 1), padding=(1, 1)),
        )

        self.bn6 = nn.BatchNorm2d(720)
        self.maxpool6 = nn.MaxPool2d((2, 2))

        ## 1st linear unit
        self.fc1 = nn.Linear(720, 400)
        self.bn7 = nn.BatchNorm1d(400)

        ## 2nd linear unit
        self.fc2 = nn.Linear(400, 180)
        self.bn8 = nn.BatchNorm1d(180)

        ## 3rd linear unit
        self.fc3 = nn.Linear(180, self.num_classes)

        ## Set a dropout

        dropout = self.cfg['dropout']
        self.drop = nn.Dropout(dropout)

        ## Use Adam optimizer
        self.optim = Adam(self.parameters(), lr=self.cfg['learning_rate'])

    def forward(self, x):
        """
        Compute forward pass for the model

        :param x: an image of size (100x100)
        """

        relu = nn.ReLU()

        ## (1) 1st hidden conv layer # 4 x 100 x 100
        x_add = self.conv11(x)
        x = self.conv12(x_add)

        x = x + x_add
        x = self.bn1(x)
        x = relu(x)
        x = self.maxpool1(x)

        ## (2) 2nd hidden conv layer # 50 x 50 x 50 -> 48 x 48
        x_add = self.conv21(x)
        x = self.conv22(x_add)

        x = x + x_add
        x = self.bn2(x)
        x = relu(x)
        x = self.maxpool2(x)

        ## (3) 3rd hidden conv layer # 125 x 24 x 24
        x_add = self.conv31(x)
        x = self.conv32(x_add)

        x = x + x_add
        x = self.bn3(x)
        x = relu(x)
        x = self.maxpool3(x)

        ## (4) 4th hidden conv layer # 250 x 12 x 12
        x_add = self.conv41(x)
        x = self.conv42(x_add)

        x = x + x_add
        x = self.bn4(x)
        x = relu(x)
        x = self.maxpool4(x)

        ## (5) 5th hidden conv layer # 420 x 6 x 6
        x_add = self.conv51(x)
        x = self.conv52(x_add)

        x = x + x_add
        x = self.bn5(x)
        x = relu(x)
        x = self.maxpool5(x)

        ## (6) 6th hidden conv layer # 540 x 3 x 3
        x_add = self.conv61(x)
        x = self.conv62(x_add)

        x = x + x_add
        x = self.bn6(x)
        x = relu(x)
        x = self.maxpool6(x)

        ## (7) 1st linear hidden layer # 720 x 1 x 1
        x = self.fc1(x.view(-1, 720))
        x = self.bn7(x)
        x = relu(x)
        x = self.drop(x)

        ## (8) 2nd linear hidden layer
        x = self.fc2(x)
        x = self.bn8(x)
        x = relu(x)
        x = self.drop(x)

        ## (9) 2nd linear hidden layer
        x = self.fc3(x)

        return x

    def train_step(self, batch, use_wandb, wandb_cfg, global_step):
        batch = to_device(batch, self.cfg['device'])

        ## Inference
        prediction = self(batch['image'])
        #prediction = torch.sigmoid(prediction)

        ## Computing losses
        losses = compute_losses(prediction, batch['target_indice'], test= False)

        ## Backpropagation
        self.zero_grad()
        losses['total_loss_train'].backward()
        self.optim.step()
        ## WandB logs, show images every N iterations
        if use_wandb:
            wandb.log(losses, step=global_step)
            if global_step > 0 and global_step % wandb_cfg['upload_every'] == 0:
                wandb.log({})
    
    def test(self, test_dataloader, use_wandb, wandb_cfg, global_step):
        self.eval()
        with torch.no_grad():

            # Perform inference on the entire test set
            loss_history = None
            for batch in test_dataloader:
                batch = to_device(batch, self.cfg['device'])
                image = batch['image']
                prediction = self(image)
                #prediction = torch.sigmoid(prediction)
                losses = compute_losses(prediction, batch['target_indice'], test= True)
                loss_history = append_losses(losses, loss_history)

            # Compute the final losses
            loss_history['total_loss_test'] = to_device(loss_history['total_loss_test'], 'cpu')
            loss_history['test_confidence'] = to_device(loss_history['test_confidence'], 'cpu')
            self.result_history.append(average_losses(loss_history))
            print()
            display_losses(self.result_history[-1])

            # Upload to WandB
            if use_wandb:
                wandb.log(set_losses_wandb(self.result_history[-1]), step=global_step)
        
        self.train()
        return self.result_history[-1]