import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Conv2D_Down(torch.nn.Module):
	def __init__(self, cfg, c_in, c_out, kernel_size, padding, stride):
		super(Conv2D_Down, self).__init__()
		if cfg.unet.training.use_norm:
			self.norm_layer = torch.nn.GroupNorm(cfg.unet.model.group_norm, c_out)
		
		self.layers = torch.nn.Sequential(
					  torch.nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
					  torch.nn.ReLU(),
					  self.norm_layer,
					  torch.nn.Conv2d(c_out, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
					  torch.nn.ReLU(),
					  self.norm_layer,
					  torch.nn.MaxPool2d(kernel_size=2)
					  )

	def forward(self, x):
		return self.layers(x)

class Conv2D(torch.nn.Module):
	def __init__(self, cfg, c_in, c_out, kernel_size=3, padding=1, stride=1):
		super(Conv2D, self).__init__()
		if cfg.unet.training.use_norm:
			self.norm_layer = torch.nn.GroupNorm(cfg.unet.model.group_norm, c_out)
		
		self.layers = torch.nn.Sequential(
					  torch.nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
					  torch.nn.ReLU(),
					  self.norm_layer)

	def forward(self, x):
		return self.layers(x)


class Encoder(nn.Module):
	def __init__(self, cfg):
		super(Encoder, self).__init__()
		self.cfg = cfg
		self.c_in = cfg.unet.model.in_channels # 3 for RGB and 1 for grayscale
		self.embd_out = cfg.unet.model.embd_dim # 768
		# input image is 3x256x256
		c_out = 32
		self.conv1 = Conv2D(cfg, self.c_in, c_out, kernel_size=5, padding=2, stride=1)
		# 32x256x256
		self.encoder = nn.ModuleList([self.conv1,
									  Conv2D_Down(cfg, c_out, c_out*2, kernel_size=3, padding=1, stride=1), # 64x128x128
									  Conv2D_Down(cfg, c_out*2, c_out*4, kernel_size=3, padding=1, stride=1), # 128x64x64
									  Conv2D_Down(cfg, c_out*4, c_out*8, kernel_size=3, padding=1, stride=1), # 256x32x32
									  Conv2D_Down(cfg, c_out*8, c_out*16, kernel_size=3, padding=1, stride=1) # 512x16x16
									  ])

		self.encode_last_layer = Conv2D(cfg, c_out*16, self.embd_out, kernel_size=3, padding=1, stride=1) # 768x16x16

	def forward(self, x):
		emd_list = []
		for layer in self.encoder:
			x = layer(x)
			emd_list.append(x)
		x = self.encode_last_layer(x)
		emd_list.append(x)
		return emd_list

class Conv2D_Up(torch.nn.Module):
	def __init__(self, cfg, c_in, c_out, kernel_size=4, padding=1, stride=2):
		super(Conv2D_Up, self).__init__()
		if cfg.unet.training.use_norm:
			self.norm_layer = torch.nn.GroupNorm(cfg.unet.model.group_norm, c_out)
		
		self.conv_layer = Conv2D(cfg, c_in*2, c_out, kernel_size=3, padding=1, stride=1)
		
		self.up_layer = torch.nn.Sequential(
					  torch.nn.ConvTranspose2d(c_out, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
					  torch.nn.ReLU(),
					  self.norm_layer)	
		
	def forward(self, x, skip):
		x = torch.cat([x, skip], 1)
		x = self.conv_layer(x)
		x = self.up_layer(x)
		return x
	
class Decoder(nn.Module):
	def __init__(self, cfg, embd_out=768, out_ch=3):
		super(Decoder, self).__init__()
		self.cfg = cfg
		self.embd_out = cfg.unet.model.embd_dim # 768
		self.out_ch = cfg.unet.model.out_channels # 3 for RGB and 1 for grayscale
		c_out = 512
		self.conv1 = Conv2D(cfg, embd_out, c_out, kernel_size=3, padding=1, stride=1) # 512x16x16
		self.decoder = nn.ModuleList([Conv2D_Up(cfg, c_out, c_out//2, kernel_size=4, padding=1, stride=2), # 256x32x32
									  Conv2D_Up(cfg, c_out//2, c_out//4, kernel_size=4, padding=1, stride=2), # 128x64x64
									  Conv2D_Up(cfg, c_out//4, c_out//8, kernel_size=4, padding=1, stride=2), # 64x128x128
									  Conv2D_Up(cfg, c_out//8, c_out//16, kernel_size=4, padding=1, stride=2) # 32x256x256
									  ])

		self.conv2 = torch.nn.Conv2d(c_out//16, self.out_ch, kernel_size=3, padding=1, stride=1) # 3x256x256 for RGB and 1x256x256 for grayscale

	
	def forward(self, embd_list):
		x = self.conv1(embd_list[-1])
		for i, layer in enumerate(self.decoder):
			x = layer(x, embd_list[-(i+2)])
		x = F.sigmoid(self.conv2(x))
		return x
	

# UNET model for image feature extraction
class UNET(nn.Module):
	def __init__(self, cfg):
		super(UNET, self).__init__()
		self.cfg = cfg
		self.Encoder = Encoder(cfg)
		self.Decoder = Decoder(cfg)
	
	def forward(self, x):
		embd_list = self.Encoder(x)
		x = self.Decoder(embd_list)
		return x
