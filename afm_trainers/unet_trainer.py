import numpy as np
import cv2
import torch
import pytorch_lightning as pl
from torchvision.utils import make_grid

from upsrt_afm.model.model import UpSRT
from dino.model.model import DINOv2KeyExtractor

from accelerate import Accelerator

from upsrt_afm.model.small_cnn import SmallCNN

from upsrt_afm.model.unet import Encoder, Decoder
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
		pred, gt = self.forward(batch)
		loss = self.loss_fn(pred, gt)
		self.log('train/loss', loss, prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		pred, gt = self.forward(batch)
		loss = self.loss_fn(pred, gt)
		if batch_idx == 0:
			num_imgs_to_show = 10
			pred = pred.detach().cpu()[:num_imgs_to_show]
			gt = gt.detach().cpu()[:num_imgs_to_show]
			grid_images = torch.cat([gt, pred], dim=0) # shape (2*batch_size, 3, 256, 256)
			captions = ["GT"] * num_imgs_to_show + ["Pred"] * num_imgs_to_show
			self.logger.experiment.log(
				{"samples": [wandb.Image(img, caption=caption) for (img, caption) in zip(grid_images, captions)]})

		self.log('val/loss', loss, prog_bar=True)
	
	def configure_optimizers(self):                         
		lr = self.cfg.unet.training.lr
		weight_decay = self.cfg.unet.training.weight_decay #self.cfg.TRAIN.L2_PENALTY
		model_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
		opt = torch.optim.Adam((model_params), lr, weight_decay=weight_decay)
			  
		return opt