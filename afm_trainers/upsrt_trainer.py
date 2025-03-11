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

from upsrt.model.unet import Encoder, Decoder

from upsrt.renderer.cameras import create_cameras_with_identity_extrinsics, get_queryCameras

from einops import rearrange
import wandb

class UpSRTTrainer(pl.LightningModule):
	def __init__(self, cfg, models, optimizer=None, loss_fn=None, Scheduler=None):
		super(UpSRTTrainer, self).__init__()
		self.cfg = cfg
		self.models = models
		self.feature_extractor = self.models['feature_extractor']
		self.UpSRT = self.models['UpSRT']
		if 'perceptual_network' in self.models:
			self.perceptual_network = self.models['perceptual_network']
		if self.cfg.upsrt.training.use_diffusion:
			self.diffusion_model = self.models['diffusion_model']
		self.optimizer = optimizer
		self.scheduler = Scheduler

		# self.loss_fn = torch.nn.MSELoss() #loss_fn
		if self.cfg.upsrt.training.modified_mse:
			self.loss_fn = self.modified_mse
		elif self.cfg.upsrt.training.perceptual_loss:
			self.loss_fn = self.perceptual_loss
		elif self.cfg.upsrt.training.mixed_loss:
			self.loss_fn = self.mixed_loss
		elif self.cfg.upsrt.training.bce_loss:
			self.loss_fn = torch.nn.BCELoss()
		else:
			self.loss_fn = torch.nn.MSELoss()
		# self.loss_fn = torch.nn.L1Loss()
		
		self.save_hyperparameters()
	
	def preprocess(self, x):
		"""
		Args:
			x   :   torch.Tensor with shape [B, C, H, W] and values in the range [-1, 1]
		"""
		unnorm_img = x * 0.5 + 0.5 # unnorm is in the range [0, 1]
		return unnorm_img
	
	def unnormalize(self, x):
		'''
		Unnormalize [-1, 1] to [0, 1]
		'''
		return torch.clip((x + 1.0) / 2.0, 0.0, 1.0)
	
	def modified_mse(self, pred, gt):
		"""
		Args:
			pred    :   torch.Tensor with shape [B, C, H, W] and values in the range [0, 1]
			gt      :   torch.Tensor with shape [B, C, H, W] and values in the range [0, 1]
		"""
		# get a mask where the gt is not [1, 1, 1] along all channels
		# print('gt shape:', gt.shape)
		# print('pred shape:', pred.shape)

		mask_ = torch.all(gt != 1.0, dim=1).float()
		# print('mask shape:', mask_.shape)
		mask = torch.repeat_interleave(mask_.unsqueeze(1), 3,  dim=1)
		mask = torch.add(mask, 1.0) # to avoid division by zero
		# print('repeated mask shape:', mask.shape)
		# mse = ((((pred - gt) ** 2) * mask).sum(dim=(1,2,3)) / mask.sum(dim=(1,2,3))).mean()
		mse = (((pred - gt) ** 2) * mask).mean()
		return mse
	
	def perceptual_loss(self, pred, gt):
		
		with torch.no_grad():
			gt_feats = self.perceptual_network(gt)[-1]
		pred_feats = self.perceptual_network(pred)[-1]    
		loss = torch.nn.functional.mse_loss(pred_feats, gt_feats)
		return loss

	def mixed_loss(self, pred, gt):
		"""
		Args:
			pred    :   torch.Tensor with shape [B, C, H, W] and values in the range [0, 1]
			gt      :   torch.Tensor with shape [B, C, H, W] and values in the range [0, 1]
		"""
		mse_loss = self.modified_mse(pred, gt)
		perceptual_loss = self.perceptual_loss(pred, gt)
		loss = mse_loss + perceptual_loss
		return loss

	def forward(self, batch):
		# prepare inputs
		input_views, query_view, R, T = batch
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
		### UpSRT model ###
		###################
		predicted_query_view, _ = self.UpSRT.get_query_features(
				dino_features=image_features, input_cameras=input_cameras,
				query_cameras=query_cameras, image_size=(256, 256),
				decoder_return_type = "pred_rgb",
				return_grid_rays = False
			)
		predicted_query_view = torch.permute(predicted_query_view, (0, 3, 1, 2)).contiguous()
		###################
		query_view = query_view.squeeze(1)
		
		return input_views, predicted_query_view, query_view
	
	def training_step(self, batch, batch_idx):   
		input_views, pred, gt = self.forward(batch)
		if torch.isnan(pred).any():
			print('Training step: Nan detected in pred')
		loss = self.loss_fn(pred, gt)
		self.log('train/loss', loss, prog_bar=True)

		if batch_idx % 100 == 0:
			num_imgs_to_show = 10
			# input_views, query_view, K, R, T = batch
			num_views = input_views.shape[1]
			# save input views
			input_views = rearrange(input_views[:num_imgs_to_show], "b n c h w -> (b n) c h w")
			input_views = input_views.detach().cpu()
			input_view_grid = make_grid(input_views, nrow=num_views)
			self.logger.experiment.log({"train/input_views": wandb.Image(input_view_grid.permute(1, 2, 0).numpy())})

			# save gt and pred images
			pred = pred.detach().cpu()[:num_imgs_to_show]
			gt = gt.detach().cpu()[:num_imgs_to_show]
			grid_images = torch.cat([gt, pred], dim=0) # shape (2*batch_size, 3, 256, 256)
			captions = ["GT"] * self.cfg.upsrt.training.batch_size + ["Pred"] * self.cfg.upsrt.training.batch_size
			self.logger.experiment.log(
				{"train/samples": [wandb.Image(img, caption=caption) for (img, caption) in zip(grid_images, captions)]})
		return loss

	def validation_step(self, batch, batch_idx):
		# wandb_logger = self.logger.experiment
		input_views, pred, gt = self.forward(batch)
		if torch.isnan(pred).any():
			print('Validation step: Nan detected in pred')
		loss = self.loss_fn(pred, gt)
		if batch_idx % 100 == 0:
			num_imgs_to_show = 10
			# input_views, query_view, K, R, T = batch
			num_views = input_views.shape[1]
			# save input views
			input_views = rearrange(input_views[:num_imgs_to_show], "b n c h w -> (b n) c h w")
			input_views = input_views.detach().cpu()
			input_view_grid = make_grid(input_views, nrow=num_views)
			self.logger.experiment.log({"input_views": wandb.Image(input_view_grid.permute(1, 2, 0).numpy())})

			# save gt and pred images
			pred = pred.detach().cpu()[:num_imgs_to_show]
			gt = gt.detach().cpu()[:num_imgs_to_show]
			grid_images = torch.cat([gt, pred], dim=0) # shape (2*batch_size, 3, 256, 256)
			captions = ["GT"] * self.cfg.upsrt.training.batch_size + ["Pred"] * self.cfg.upsrt.training.batch_size
			self.logger.experiment.log(
				{"samples": [wandb.Image(img, caption=caption) for (img, caption) in zip(grid_images, captions)]})
		# if batch_idx == 0:
		#     num_imgs_to_show = 10
		#     pred = pred.detach().cpu()[:num_imgs_to_show]
		#     gt = gt.detach().cpu()[:num_imgs_to_show]
		#     grid_images = torch.cat([gt, pred], dim=0) # shape (2*batch_size, 3, 256, 256)
		#     captions = ["GT"] * self.cfg.upsrt.training.batch_size + ["Pred"] * self.cfg.upsrt.training.batch_size
		#     self.logger.experiment.log(
		#         {"samples": [wandb.Image(img, caption=caption) for (img, caption) in zip(grid_images, captions)]})

		self.log('val/loss', loss, prog_bar=True, sync_dist=False)
		return loss

	def test_step(self, batch, batch_idx):
		input_views, pred, gt = self.forward(batch)
		print('pred:', torch.min(pred), torch.max(pred), torch.unique(pred))    
		if torch.isnan(pred).any():
			print('Test step: Nan detected in pred')
		
		loss = self.loss_fn(pred, gt)
		print('loss:', loss)
		# save gt and pred images
		# cv2.imwrite(f"gt_{batch_idx}.png", gt[0].permute(1, 2, 0).cpu().numpy())
		# cv2.imwrite(f"pred_{batch_idx}.png", pred[0].permute(1, 2, 0).cpu().numpy())
		# log images to wandb
		wandb.log({"gt": wandb.Image(gt[0].permute(1, 2, 0).cpu().numpy()),
				   "pred": wandb.Image(pred[0].permute(1, 2, 0).cpu().numpy())})

		return loss
	
	def configure_optimizers(self):                         
		lr = self.cfg.upsrt.training.lr
		weight_decay = self.cfg.upsrt.training.weight_decay #self.cfg.TRAIN.L2_PENALTY
		if self.cfg.upsrt.training.use_dino:
			model_params = list(self.UpSRT.parameters())
		else:
			model_params = list(self.feature_extractor.parameters()) + list(self.UpSRT.parameters())

		opt = torch.optim.Adam((model_params), lr, weight_decay=weight_decay)
		return opt   
