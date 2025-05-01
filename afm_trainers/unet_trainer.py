import numpy as np
import cv2
import torch
import pytorch_lightning as pl
from torchvision.utils import make_grid

from upsrt.model.model import UpSRT
from dino.model.model import DINOv2KeyExtractor

from accelerate import Accelerator

from upsrt.model.small_cnn import SmallCNN

from upsrt.model.unet import Encoder, Decoder
from einops import rearrange
import wandb

class UNetTrainer(pl.LightningModule):
	def __init__(self, cfg, optimizer=None, loss_fn=None, Scheduler=None):
		super(UNetTrainer, self).__init__()
		self.cfg = cfg
		self.encoder = Encoder(cfg)
		self.decoder = Decoder(cfg)
		self.optimizer = optimizer
		self.scheduler = Scheduler
		self.loss_fn = torch.nn.MSELoss() #loss_fn
		# self.loss_fn = torch.nn.L1Loss()
		
		self.save_hyperparameters()
	
	def forward(self, inp_img):
		self.embd_list = self.encoder(inp_img)
		pred = self.decoder(self.embd_list)

		return pred, inp_img
	
	def training_step(self, batch, batch_idx):  
		preds, gts = self.forward(batch)
		loss = self.loss_fn(preds, gts)
		self.log('train/loss', loss, prog_bar=True)
		if self.cfg.data.use_depth:
			pred = preds[:, :3, :, :] # shape (batch_size, 3, 256, 256)
			gt = gts[:, :3, :, :] # shape (batch_size, 3, 256, 256)
			depth_pred = preds[:, 3, :, :].unsqueeze(1) # shape (batch_size, 1, 256, 256)
			depth_gt = gts[:, 3, :, :].unsqueeze(1) # shape (batch_size, 1, 256, 256)
		else:
			pred = preds
			gt = gts
		if batch_idx % 100 == 0:
			num_imgs_to_show = 10
			pred = pred.detach().cpu()[:num_imgs_to_show]
			gt = gt.detach().cpu()[:num_imgs_to_show]
			if self.cfg.data.use_depth:
				depth_pred = depth_pred.detach().cpu()[:num_imgs_to_show]
				depth_gt = depth_gt.detach().cpu()[:num_imgs_to_show]
				grid_images = torch.cat([gt, pred], dim=0) # shape (4*batch_size, 4, 256, 256)
				captions = ["GT"] * num_imgs_to_show + ["Pred"] * num_imgs_to_show 
				grid_imaged_depth = torch.cat([depth_gt, depth_pred], dim=0) # shape (2*batch_size, 1, 256, 256)
				captions_depth = ["GT Depth"] * num_imgs_to_show + ["Pred Depth"] * num_imgs_to_show
			else:
				grid_images = torch.cat([gt, pred], dim=0) # shape (2*batch_size, 3, 256, 256)
				captions = ["GT"] * num_imgs_to_show + ["Pred"] * num_imgs_to_show
			self.logger.experiment.log(
				{"train/samples": [wandb.Image(img, caption=caption) for (img, caption) in zip(grid_images, captions)]})
			if self.cfg.data.use_depth:
				self.logger.experiment.log(
					{"train/depth_samples": [wandb.Image(img, caption=caption) for (img, caption) in zip(grid_imaged_depth, captions_depth)]})
		return loss

	def validation_step(self, batch, batch_idx):
		preds, gts = self.forward(batch)
		loss = self.loss_fn(preds, gts)
		if self.cfg.data.use_depth:	
			pred = preds[:, :3, :, :] # shape (batch_size, 3, 256, 256)
			gt = gts[:, :3, :, :] # shape (batch_size, 3, 256, 256)
			depth_pred = preds[:, 3, :, :].unsqueeze(1) # shape (batch_size, 1, 256, 256)
			depth_gt = gts[:, 3, :, :].unsqueeze(1) # shape (batch_size, 1, 256, 256)
		else:
			pred = preds
			gt = gts
		if batch_idx % 100 == 0:
			num_imgs_to_show = 10
			pred = pred.detach().cpu()[:num_imgs_to_show]
			gt = gt.detach().cpu()[:num_imgs_to_show]
			if self.cfg.data.use_depth:
				depth_pred = depth_pred.detach().cpu()[:num_imgs_to_show]
				depth_gt = depth_gt.detach().cpu()[:num_imgs_to_show]
				grid_images = torch.cat([gt, pred], dim=0) # shape (4*batch_size, 4, 256, 256)
				captions = ["GT"] * num_imgs_to_show + ["Pred"] * num_imgs_to_show 
				grid_imaged_depth = torch.cat([depth_gt, depth_pred], dim=0) # shape (2*batch_size, 1, 256, 256)
				captions_depth = ["GT Depth"] * num_imgs_to_show + ["Pred Depth"] * num_imgs_to_show
			else:
				grid_images = torch.cat([gt, pred], dim=0) # shape (2*batch_size, 3, 256, 256)
				captions = ["GT"] * num_imgs_to_show + ["Pred"] * num_imgs_to_show
			self.logger.experiment.log(
				{"val/samples": [wandb.Image(img, caption=caption) for (img, caption) in zip(grid_images, captions)]})
			if self.cfg.data.use_depth:
				self.logger.experiment.log(
					{"val/depth_samples": [wandb.Image(img, caption=caption) for (img, caption) in zip(grid_imaged_depth, captions_depth)]})
			

		self.log('val/loss', loss, prog_bar=True)
	
	def configure_optimizers(self):                         
		lr = self.cfg.unet.training.lr
		weight_decay = self.cfg.unet.training.weight_decay #self.cfg.TRAIN.L2_PENALTY
		model_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
		opt = torch.optim.Adam((model_params), lr, weight_decay=weight_decay)
			  
		return opt