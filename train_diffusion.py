import os
import numpy as np
from omegaconf import OmegaConf
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from data.data import UpsrtDataset


# from upsrt.model.model_afm import UpSRT
from dino.model.model import DINOv2KeyExtractor
from diffusion.pipeline_control_net import DiffusionPipelineCN
from upsrt.model.small_cnn import SmallCNN
from upsrt.model.model import UpSRT
from afm_trainers.upsrt_trainer import UpSRTTrainer
from afm_trainers.unet_trainer import UNetTrainer
from afm_trainers.diffusion_trainer import DiffusionTrainer
from pytorch_lightning.plugins.environments import SLURMEnvironment

def train(args):
    # pl.seed_everything(1)
    cfg = get_cfg('./configs/training.yaml')
    cfg = update_cfg(cfg, args)
    print('#'*10, 'Config', '#'*10)
    print(OmegaConf.to_yaml(cfg))
    print('#'*30)
    # Load data
    train_data = UpsrtDataset(cfg, data_path=cfg.data.path, n_views=cfg.data.n_views, split='train')  
    val_data = UpsrtDataset(cfg, data_path=cfg.data.path, n_views=cfg.data.n_views, split='val') 
    # data loader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.diffusion.training.batch_size, shuffle=True, num_workers=cfg.diffusion.training.num_workers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=cfg.diffusion.training.batch_size, shuffle=False, num_workers=cfg.diffusion.training.num_workers, drop_last=True)

    # Load models
    # Load UpSRT model
    srt_model = UpSRT(cfg.upsrt.model)
    print('Loaded UpSRT model')
    # Load feature extractor model: DINOV2 or UNET or SmallCNN
    if cfg.upsrt.training.use_dino:    
        dino_model = DINOv2KeyExtractor(cfg.dino.model)
        print('Loaded DINO model')
        if cfg.upsrt.training.freeze_feature_extractor:
            dino_model.eval()
            for param in dino_model.parameters():
                param.requires_grad = False
            print('Set DINO model as non-trainable')
        feature_extractor = dino_model
    
    elif cfg.upsrt.training.pretrained_feature_extractor:
        if cfg.data.grayscale:
            unet = UNetTrainer.load_from_checkpoint(args.fe_grayscale_weights_dir, cfg=cfg)
        else:
            unet = UNetTrainer.load_from_checkpoint(args.fe_weights_dir, cfg=cfg)

        if cfg.upsrt.training.freeze_feature_extractor:    
            # unet.eval()
            unet.freeze()
            print('Set UNET model as non-trainable')
        feature_extractor = unet.encoder    
        print('Loaded pre-trained weights for feature extractor UNET model')
        
    else:
        if cfg.data.grayscale:
            feature_extractor = SmallCNN(in_dim=1, out_dim=768)
        else:
            feature_extractor = SmallCNN(in_dim=3, out_dim=768)
        print('Using SmallCNN model for feature extraction')

    # load checkpoint for UpSRT model
    pretrained_upsrt_ckpt_path = cfg.diffusion.training.pretrained_upsrt_ckpt_path
    ckpt = torch.load(pretrained_upsrt_ckpt_path, map_location='cpu')
    # load appropriate checkpoint for UpSRT model and feature extractor
    srt_model.load_state_dict({k.replace('UpSRT.', ''): v for k, v in ckpt['state_dict'].items() if 'UpSRT' in k})
    print('Loaded UpSRT model with pretrained weights')

    feature_extractor.load_state_dict({k.replace('feature_extractor.', ''): v for k, v in ckpt['state_dict'].items() if 'feature_extractor' in k})
    print('Loaded feature extractor model with pretrained weights')

    # freeze feature extractor and UpSRT
    feature_extractor.eval()
    for param in feature_extractor.parameters():
        param.requires_grad = False
    print('Set feature extractor as non-trainable')

    srt_model.eval()
    for param in srt_model.parameters():
        param.requires_grad = False
    print('Set UpSRT model as non-trainable')

    # Load diffuision model 
    diffusion_pipeline = DiffusionPipelineCN(
        cfg.diffusion.model, srt_model=srt_model,
        dino_model=dino_model
    )
    print('Loaded Diffusion model')
    if cfg.diffusion.training.use_pretrained_upfusion:
        diffusion_load_dict = torch.load(cfg.diffusion.training.pretrained_upfusion_ckpt_path, map_location="cpu")
        diffusion_pipeline.handle.load_state_dict(diffusion_load_dict['model_state_dict'])
        print('Loaded pretrained UpFusion weights for diffusion model')
    if cfg.diffusion.training.use_pretrained_diffusion:    
        sdv15_ini_weights = torch.load(cfg.diffusion.training.pretrained_sd15_ckpt_path, map_location="cpu")
        diffusion_pipeline.handle.load_state_dict(sdv15_ini_weights)
        print('Loaded pretrained SDv1.5 weights for diffusion model')
    
    models = {'feature_extractor': feature_extractor, 'UpSRT': srt_model, 'diffusion_model': diffusion_pipeline}
    
    Diffusion_model = DiffusionTrainer(cfg, models)
    
    # Initiate the trainer
    wandb_logger = pl.loggers.WandbLogger(name=cfg.diffusion.training.logging.exp_name,
                                        project=cfg.diffusion.training.logging.project_name, dir=cfg.diffusion.training.logging.save_dir,)

    # tensorboard_logger = pl.loggers.TensorBoardLogger(cfg.diffusion.training.logging.save_dir, name=cfg.diffusion.training.logging.exp_name)

    monitor_val = 'val/loss'
    
       
    checkpoint = ModelCheckpoint(monitor=monitor_val,
                                dirpath=cfg.diffusion.training.logging.save_dir+'/'+cfg.diffusion.training.logging.exp_name, 
                                filename='{epoch}-{step}',
                                mode='min', 
                                save_last=True)

    trainer = pl.Trainer(devices=cfg.diffusion.training.gpus, 
                        num_nodes=cfg.diffusion.training.num_nodes,
                        accelerator='gpu', 
                        strategy=DDPStrategy(find_unused_parameters=False, static_graph=True),#'ddp',
                        #plugins=DDPPlugin(find_unused_parameters=False),
                        # plugins=[SLURMEnvironment(auto_requeue=False)], # need to comment this if not using slurm
                        callbacks=[checkpoint],
                        logger=[wandb_logger],
                        max_epochs=cfg.diffusion.training.epochs, 
                        default_root_dir=cfg.diffusion.training.logging.save_dir+'/'+cfg.diffusion.training.logging.exp_name, 
                        fast_dev_run=args.debug_run,
                        log_every_n_steps=50)

    if not args.test: 
        print('#'*100, 'Training', '#'*100)
        trainer.fit(Diffusion_model, train_loader, val_loader, ckpt_path=None)
    else:
        test_set = torch.utils.data.Subset(val_data, range(10))
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=cfg.diffusion.training.num_workers)
        # ckpt_path = './training_logs_upsrt_finetune/bs4_ep_100_data100k_upsrt_train_from_scratch_and_unet_freezed_unet_without_plucker_coordinates_perceptual_loss/epoch=4-step=31250.ckpt'
        ckpt_path = './training_logs_upsrt_finetune/bs4_ep_100_data100k_upsrt_train_from_scratch_and_unet_freezed_unet_without_plucker_coordinates_perceptual_loss_ndc_from_minus1_to_1/last.ckpt'
        ckpt = torch.load(ckpt_path, map_location='cuda')
        print('######## Loaded checkpoint ########')
        print(ckpt['state_dict'].keys())
        trainer.test(Diffusion_model, dataloaders=test_loader, ckpt_path=ckpt_path)

def get_cfg(cfg_path, verbose=False):
    cfg = OmegaConf.load(cfg_path)
    if verbose:
        print(OmegaConf.to_yaml(cfg))
    return cfg

def update_cfg(cfg, args):
    # Data related
    cfg.data.num_samples = args.num_samples
    cfg.data.grayscale = args.grayscale
    if args.grayscale_3ch:
        cfg.data.grayscale_3ch = True
    cfg.data.n_views = args.n_views
    cfg.data.fixed_views = args.fixed_views
    if cfg.data.fixed_views:
        cfg.data.n_views = 6
    if args.white_background:
        cfg.data.white_background = True
    if args.return_rays:
        cfg.data.return_rays = True
    if args.identity_K:
        cfg.data.identity_K = True
    # Upsrt related
    cfg.upsrt.model.ray.parameterize = args.ray_parameterize
    cfg.upsrt.training.use_dino = args.use_dino
    cfg.upsrt.training.pretrained_upsrt = args.pretrained_upsrt
    cfg.upsrt.training.pretrained_feature_extractor = args.pretrained_feature_extractor
    cfg.upsrt.training.freeze_feature_extractor = args.freeze_feature_extractor
    cfg.upsrt.training.freeze_upsrt_encoder = args.freeze_upsrt_encoder
    cfg.upsrt.training.freeze_upsrt = args.freeze_upsrt
    if cfg.data.grayscale:
        cfg.upsrt.model.ray_decoder.grayscale = args.grayscale
    
    # Diffusion related
    cfg.diffusion.training.num_workers = args.num_workers
    cfg.diffusion.training.epochs = args.epochs
    cfg.diffusion.training.num_nodes = args.num_nodes
    cfg.diffusion.training.gpus = args.gpus
    cfg.diffusion.training.batch_size = args.batch_size
    cfg.diffusion.training.lr = args.lr
    cfg.diffusion.training.weight_decay = args.weight_decay
    if args.use_pretrained_diffusion:
        cfg.diffusion.training.use_pretrained_diffusion = True
    if args.use_pretrained_upfusion:
        cfg.diffusion.training.use_pretrained_upfusion = True
    if args.project_name:
        cfg.diffusion.training.logging.project_name = args.project_name
    if args.exp_name:
        cfg.diffusion.training.logging.exp_name = args.exp_name
    if args.save_dir:
        cfg.diffusion.training.logging.save_dir = args.save_dir
    
    if args.sd_locked:
        cfg.diffusion.model.control_net_sd_locked = True
    # If using U-Net for feature extraction with grayscale images
    if args.grayscale:
        cfg.data.grayscale = True
        cfg.unet.model.in_channels = 1
        cfg.unet.model.out_channels = 1


    return cfg

if __name__ == '__main__':
   args = argparse.ArgumentParser()
   args.add_argument('--save_dir', type=str, help='Update in the config.yaml file')
   args.add_argument('--num_samples', type=str, default='1k', help='number of samples to train from [256, 1k, 10k, 50k, 100k, whole_data]')
   args.add_argument('--num_nodes', type=int, default=1)
   args.add_argument('--gpus', type=int, default=1)
   args.add_argument('--batch_size', type=int, default=8, help='local batch size per gpu for training')
   args.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
   args.add_argument('--weight_decay', type=float, default=0.0, help='weight_decay for regularization term in optimizer')
   args.add_argument('--num_workers', type=int, default=4)
   args.add_argument('--epochs', type=int, default=10)
   args.add_argument('--project_name', type=str, help='Update in the config.yaml file')
   args.add_argument('--exp_name', type=str, help='Update in the config.yaml file')
   args.add_argument('--use_dino', action='store_true', help='use dino model for feature extraction')
   args.add_argument('--pretrained_upsrt', action='store_true', help='use pretrained upsrt model for training')
   args.add_argument('--pretrained_feature_extractor', action='store_true', help='use pretrained encoder for feature extraction')
   args.add_argument('--freeze_feature_extractor', action='store_true', help='freeze the layers of feature extractor')
   args.add_argument('--freeze_upsrt_encoder', action='store_true', help='freeze Encoder layers of UpSRT model')
   args.add_argument('--freeze_upsrt', action='store_true', help='freeze All layers of UpSRT model')
   args.add_argument('--ray_parameterize', type=str, help='plucker/no_plucker', default='plucker')
   args.add_argument('--grayscale', action='store_true', help='use grayscale images')
   args.add_argument('--test', action='store_true', help='test the model')
   args.add_argument('--n_views', type=int, default=3, help='Number of input views')
   args.add_argument('--fixed_views', action='store_true', help='If use 6 fixed orthogonal views')
   args.add_argument('--use_diffusion', action='store_true', help='Use diffusion model for predecting novel views')
   args.add_argument('--return_rays', action='store_true', help='If True return rays directly instead of camera parameters in Dataloader')
   args.add_argument('--debug_run', action='store_true', help='Pytorch Lightning debug run for testing')
   args.add_argument('--use_pretrained_diffusion', action='store_true', help='Initialize diffusion model with  pretrained SDv1.5 weights')
   args.add_argument('--use_pretrained_upfusion', action='store_true', help='Initialize diffusion model with  pretrained UpFusion weights')
   args.add_argument('--sd_locked', action='store_true', help='Lock the training of SD decoder blocks')
   args.add_argument('--identity_K', action='store_true', help='If True Use Idnetity matrix as K matrix for query rays/cameras')
   args.add_argument('--grayscale_3ch', action='store_true', help='use grayscale with 3ch images')
   args.add_argument('--white_background', action='store_true', help='use white background')
   args = args.parse_args()
   
   train(args)
   