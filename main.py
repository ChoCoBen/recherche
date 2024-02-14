from argparse import ArgumentParser
import torch
from src.utils.helper import *
import os
import time
from src.core.model import ConvModel

## Set up the paser
parser = ArgumentParser()
parser.add_argument('--config', help='path to experiment config file')
parser.add_argument('--wandb-config', help='path to wandb config file')
parser.add_argument('--checkpoint', help='path to checkpoint file')
args = parser.parse_args()

## Load general config from yaml file, some global variables
cfg = load_config(args.config)
num_epochs = cfg['num_epochs']

## Set the hardware device to run the model on
device = setup_device(cfg)

## Random seed
torch.random.manual_seed(cfg['seed'])

## Create the dataset object
train_dataloader, test_dataloader = setup_dataloaders(cfg)

## Initialize the model
model = ConvModel(cfg).to(cfg['device'])

## If a checkpoint is specified, load it
if args.checkpoint is not None:
    model.load_checkpoint(args.checkpoint)

## Wandb setup
use_wandb = args.wandb_config is not None
if use_wandb:
    wandb_cfg = load_config(args.wandb_config)
    setup_wandb(cfg, wandb_cfg, model)
else: wandb_cfg = None

## Time measurement
absolute_start = time.time()
start = absolute_start

## Training loop
print(f'[INFO] {cfg["batch_size"]} gestures per bach, {len(train_dataloader)} batches per epoch, {cfg["num_epochs"]} epochs.')
print(f'[INFO] {num_epochs*len(train_dataloader)} batches total')

global_step = 0
do_save_test_history_result = False

for epoch in range(0, num_epochs):
    print(f'[INFO] EPOCH {epoch}')

    for batch in train_dataloader:
        print(f'[INFO] Processing batch {str(global_step+1).ljust(10)}', end='')

        model.train_step(batch, use_wandb, wandb_cfg, global_step)

        if cfg['test_every'] > 0 and global_step % cfg['test_every'] == 0: #can add: 'and global_step > 0'
            result = model.test(test_dataloader, use_wandb, wandb_cfg, global_step)
            isbest = save_result_model(result, model, cfg)
            
            if isbest:
                do_save_test_history_result = True

        # Update the number of batches done
        global_step += 1

        # end of the batch backprop, timing
        stop = time.time()
        delay = stop - start
        t1 = stop - absolute_start
        t2 = 1/(delay)
        t3 = (num_epochs*len(train_dataloader)-global_step)*delay

        print(f'{print_time(t1).rjust(9)} elapsed | {t2:.2f} it/s | {print_time(t3).rjust(9)} left', end='\r')
        start = stop

## Linebreak for readability
print()

# perform inference on test set to evaluate
result = model.test(test_dataloader, use_wandb, wandb_cfg, global_step)
save_result_model(result, model, cfg)

if do_save_test_history_result:
    save_history(model.result_history, cfg)
