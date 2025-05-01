import os
import numpy as np
from omegaconf import OmegaConf
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy, ParallelStrategy, FSDPStrategy #ParrallelStrategy doesnt work

from data.data import UpsrtDataset, AutoEncoderDataset

from afm_trainers.unet_trainer import UNetTrainer
from pytorch_lightning.plugins.environments import SLURMEnvironment
# from lightning.pytorch.plugins.environments import SLURMEnvironment
# SEED = np.random.randint(0, 1000)


def train(args):
    # pick random seed
    # pl.seed_everything(SEED)
    
    cfg = get_cfg('./configs/training.yaml')
    # cfg.unet.training.seed = SEED
    cfg = update_cfg(cfg, args)
    print('#'*10, 'Config', '#'*10)
    print(OmegaConf.to_yaml(cfg))
    print('#'*30)
    # Load data
    train_data = AutoEncoderDataset(cfg, data_path=cfg.data.path, split='train')  
    val_data = AutoEncoderDataset(cfg, data_path=cfg.data.path, split='val') 
    # data loader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.unet.training.batch_size, shuffle=True, pin_memory=True, num_workers=cfg.unet.training.num_workers)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=cfg.unet.training.batch_size, shuffle=False, pin_memory=True, num_workers=cfg.unet.training.num_workers)

    # Load model
    unet_model = UNetTrainer(cfg)
    # Initiate the trainer
    # logger = pl.loggers.TensorBoardLogger(cfg.DIR.OUT_PATH, name=cfg.DIR.EXPERIMENT_NAME)

    wandb_logger = pl.loggers.WandbLogger(name=cfg.unet.training.logging.exp_name,
                                        project=cfg.unet.training.logging.project_name, dir=cfg.unet.training.logging.save_dir,)

    monitor_val = 'val/loss'
    
    checkpoint = ModelCheckpoint(monitor=monitor_val,
                                dirpath=cfg.unet.training.logging.save_dir+'/'+cfg.unet.training.logging.exp_name, 
                                filename='{epoch}-{step}',
                                mode='min', 
                                save_top_k=1,
                                save_last=True)

    trainer = pl.Trainer(devices=cfg.unet.training.gpus, 
                        num_nodes=cfg.unet.training.num_nodes,
                        accelerator='gpu', 
                        # strategy=FSDPStrategy(),#'ddp',
                        strategy=DDPStrategy(find_unused_parameters=False, static_graph=True),#'ddp',
                        # plugins=DDPPlugin(find_unused_parameters=False),
                        plugins=[SLURMEnvironment(auto_requeue=False)], # use when running on slurm
                        callbacks=[checkpoint],
                        logger=[wandb_logger], 
                        max_epochs=cfg.unet.training.epochs, 
                        default_root_dir=cfg.unet.training.logging.save_dir+'/'+cfg.unet.training.logging.exp_name, 
                        fast_dev_run=args.debug_run,
                        log_every_n_steps=50)

    trainer.fit(unet_model, train_loader, val_loader, ckpt_path=None)
                
def get_cfg(cfg_path, verbose=False):
    cfg = OmegaConf.load(cfg_path)
    if verbose:
        print(OmegaConf.to_yaml(cfg))
    return cfg

def update_cfg(cfg, args):
    
    cfg.data.num_samples = args.num_samples

    if args.grayscale:
        cfg.data.grayscale = True
        cfg.unet.model.in_channels = 1
        cfg.unet.model.out_channels = 1
    if args.grayscale_3ch:
        cfg.data.grayscale_3ch = True
        cfg.unet.model.in_channels = 3
        cfg.unet.model.out_channels = 3
    if args.use_depth:
        cfg.data.use_depth = True
        cfg.unet.model.in_channels = 4
        cfg.unet.model.out_channels = 4
    cfg.unet.training.num_workers = args.num_workers
    cfg.unet.training.epochs = args.epochs
    cfg.unet.training.num_nodes = args.num_nodes
    cfg.unet.training.gpus = args.gpus
    cfg.unet.training.batch_size = args.batch_size
    cfg.unet.training.lr = args.lr
    cfg.unet.training.weight_decay = args.weight_decay

    if args.project_name:
        cfg.unet.training.logging.project_name = args.project_name
    if args.exp_name:
        cfg.unet.training.logging.exp_name = args.exp_name
    return cfg

if __name__ == '__main__':
   args = argparse.ArgumentParser()
   args.add_argument('--weights_dir', type=str, default='./weights')
   args.add_argument('--num_samples', type=str, default='1k', help='number of samples to train from [256, 1k, 10k, 50k, 100k, whole_data]')
   args.add_argument('--num_nodes', type=int, default=1)
   args.add_argument('--gpus', type=int, default=1)
   args.add_argument('--batch_size', type=int, default=8, help='local batch size per gpu for training')
   args.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
   args.add_argument('--weight_decay', type=float, default=0.0, help='weight_decay for regularization term in optimizer')
   args.add_argument('--num_workers', type=int, default=4)
   args.add_argument('--epochs', type=int, default=10)
   args.add_argument('--save_dir', type=str, default='./training_logs_unet')
   args.add_argument('--project_name', type=str, help='Update in the unet_config.yaml file')
   args.add_argument('--exp_name', type=str, default='upsrt_unet', help='experiment name for logging')
   args.add_argument('--grayscale', action='store_true', help='use grayscale images')
   args.add_argument('--debug_run', action='store_true', help='Debug run to test the code')
   args.add_argument('--grayscale_3ch', action='store_true', help='use grayscale with 3ch images')
   args.add_argument('--use_depth', action='store_true', help='use depth values in nm as extra input channel')
   
   args = args.parse_args()
   
   train(args)
   