import numpy as np
import cv2
import torch
import pytorch_lightning as pl
from torchvision.utils import make_grid

from upsrt.model.model import UpSRT
from dino.model.model import DINOv2KeyExtractor

from accelerate import Accelerator
from upsrt.model.small_cnn import SmallCNN
from upsrt.renderer.cameras import create_cameras_with_identity_extrinsics, get_queryCameras

from einops import rearrange
import wandb

class DiffusionTrainer(pl.LightningModule):
	def __init__(self, cfg, models, optimizer=None, loss_fn=None, Scheduler=None):
		super(DiffusionTrainer, self).__init__()
		self.cfg = cfg
		self.models = models
		self.feature_extractor = self.models['feature_extractor']
		self.UpSRT = self.models['UpSRT']
		self.diffusion_model = self.models['diffusion_model']
		self.optimizer = optimizer
		self.scheduler = Scheduler
		self.loss_fn = torch.nn.MSELoss()

	def normalize(self, x):
		'''
		Normalize [0, 1] to [-1, 1]
		'''
		return torch.clip(x*2 - 1.0, -1.0, 1.0)
	
	def unnormalize(self, x):
		'''
		Unnormalize [x_min, x_max] to [0, 1]
		or
		Unnormalize [-1, 1] to [0, 1]
		'''
		# x_min = torch.min(x)
		# x_max = torch.max(x)
		# return (x - x_min) / (x_max - x_min)
		return torch.clip((x + 1.0) / 2.0, 0.0, 1.0)

	def forward(self, batch, batch_idx, test=None):
		# prepare inputs
		input_views, query_view, R, T = batch
		query_view = query_view.squeeze(1)
		query_cameras = get_queryCameras( device=self.device, min_x=-1.1, max_x=1.1, min_y=-1.1, max_y=1.1, znear=-1.1, zfar=1.1, R=R, T=T, image_size=self.cfg.data.image_size)
		input_cameras = create_cameras_with_identity_extrinsics(num_cameras=input_views.shape[1])
		input_cameras = [input_cameras] * len(input_views)
		# Feature extraction from DINO or pretrained feature extractor
		if self.cfg.upsrt.training.use_dino:
			if self.cfg.upsrt.training.freeze_feature_extractor:    
				with torch.no_grad():
					image_features = self.feature_extractor(input_views) # unnormalizaion is done inside the DINO model code
			else:
				image_features = self.feature_extractor(input_views)

		elif self.cfg.upsrt.training.pretrained_feature_extractor:
			if self.cfg.upsrt.training.freeze_feature_extractor:    
				with torch.no_grad():
					N = input_views.shape[1]
					reshaped = rearrange(input_views, "b n c h w -> (b n) c h w")
					image_features_ = self.feature_extractor(self.preprocess(reshaped))[-1]
					image_features = rearrange(image_features_, "(b n) c h w -> b n (h w) c", n=N) # (b, n, 256, 768)
			else:
				N = input_views.shape[1]
				reshaped = rearrange(input_views, "b n c h w -> (b n) c h w")
				image_features_ = self.feature_extractor(self.preprocess(reshaped))[-1]
				image_features = rearrange(image_features_, "(b n) c h w -> b n (h w) c", n=N) # (b, n, 256, 768)
		else:
			image_features = self.feature_extractor(self.preprocess(input_views))
		###################
		with torch.no_grad():	
			query_features, plucker_encoding, slt = self.UpSRT.get_query_features(
					dino_features=image_features, input_cameras=input_cameras,
					query_cameras=query_cameras, image_size=(32, 32),
					decoder_return_type = "pre_rgb_mlp_features",
					return_grid_rays = True, return_slt = True
				)
		cond_images = torch.cat((query_features, plucker_encoding), dim = 3)
		srt_cond = (cond_images, slt, query_cameras) # DF+SLT Conditioning!

		loss = self.diffusion_model.forward_with_loss(clean_data=self.normalize(query_view), srt_cond=srt_cond, cfg_seed=123,
												 enable_cfg=self.cfg.diffusion.training.enable_condition_dropout, condition_dropout=self.cfg.diffusion.training.condition_dropout_frac)
		if batch_idx % 100 ==0 or test is not None:
			decoded, _ = self.diffusion_model.forward_multi_step_denoise(clean_data=self.normalize(query_view), srt_cond=srt_cond, batch_size=input_views.shape[0], unconditional_guidance_scale=self.cfg.diffusion.model.unconditional_guidance_scale, ddim_steps=30, t_start=1, t_end=29)
			decoded = self.unnormalize(decoded).clip(0.0, 1.0)
			return loss, decoded, query_view
		else:
			return loss
		
	def training_step(self, batch, batch_idx):   
		if batch_idx % 100 == 0:
			loss, decoded, query_view = self.forward(batch, batch_idx)
			num_imgs_to_show = 10 if self.cfg.diffusion.training.batch_size > 10 else self.cfg.diffusion.training.batch_size
			decoded = decoded.detach().cpu()[:num_imgs_to_show]
			query_view = query_view.detach().cpu()[:num_imgs_to_show]
			grid_images = torch.cat([query_view, decoded], dim=0)
			captions = ["GT"] * num_imgs_to_show + ["Pred"] * num_imgs_to_show
			self.logger.experiment.log(
				{"train/samples": [wandb.Image(img, caption=caption) for (img, caption) in zip(grid_images, captions)]})
		else:
			loss = self.forward(batch, batch_idx)
		self.log('train/loss', loss, prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		#save decoded and query_view images
		if batch_idx % 100 == 0:
			loss, decoded, query_view = self.forward(batch, batch_idx)
			num_imgs_to_show = 10 if self.cfg.diffusion.training.batch_size > 10 else self.cfg.diffusion.training.batch_size
			decoded = decoded.detach().cpu()[:num_imgs_to_show]
			query_view = query_view.detach().cpu()[:num_imgs_to_show]
			grid_images = torch.cat([query_view, decoded], dim=0)
			captions = ["GT"] * num_imgs_to_show + ["Pred"] * num_imgs_to_show
			self.logger.experiment.log(
				{"samples": [wandb.Image(img, caption=caption) for (img, caption) in zip(grid_images, captions)]})
		else:
			loss = self.forward(batch, batch_idx)
		self.log('val/loss', loss, prog_bar=True)
		
		return loss
	
	def test_step(self, batch, batch_idx):
		loss, decoded, query_view = self.forward(batch, batch_idx, test=True)
		num_imgs_to_show = 10 if self.cfg.diffusion.training.batch_size > 10 else self.cfg.diffusion.training.batch_size
		decoded = decoded.detach().cpu()[:num_imgs_to_show]
		query_view = query_view.detach().cpu()[:num_imgs_to_show]
		grid_images = torch.cat([query_view, decoded], dim=0)
		captions = ["GT"] * num_imgs_to_show + ["Pred"] * num_imgs_to_show
		self.logger.experiment.log(
			{"samples": [wandb.Image(img, caption=caption) for (img, caption) in zip(grid_images, captions)]})
		self.log('test/loss', loss, prog_bar=True)
		return loss

	def configure_optimizers(self):
		lr = self.cfg.diffusion.training.lr
		weight_decay = self.cfg.diffusion.training.weight_decay #self.cfg.TRAIN.L2_PENALTY
		# model_params = list(self.diffusion_model.parameters())
		model_params = list(self.diffusion_model.handle.control_model.parameters())
		if not self.cfg.diffusion.model.control_net_sd_locked:
			model_params += list(self.diffusion_model.handle.model.diffusion_model.output_blocks.parameters())
			model_params += list(self.diffusion_model.handle.model.diffusion_model.out.parameters())
		# print number of trainable parameters
		num_params = sum(p.numel() for p in model_params if p.requires_grad)
		print('#'*50)
		print(f'Number of trainable parameters: {num_params}')
		print('#'*50)
		opt = torch.optim.Adam((model_params), lr, weight_decay=weight_decay)
		return opt   

