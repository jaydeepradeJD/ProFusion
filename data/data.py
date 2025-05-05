import os
import math
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from pytorch3d.renderer import FoVPerspectiveCameras, PerspectiveCameras, OrthographicCameras, look_at_view_transform
from pytorch3d.transforms import matrix_to_axis_angle, matrix_to_euler_angles, euler_angles_to_matrix

from upsrt.renderer.rays import (
    get_grid_rays, get_patch_rays,
    get_plucker_parameterization,
    get_random_query_pixel_rays,
    positional_encoding, get_grid_rays_gpu
)

class UpsrtDataset(Dataset):
    def __init__(self, cfg, data_path, n_views, n_query_views=1, image_size=[256, 256], split='train', transform=None):
        self.cfg = cfg
        self.data_path = data_path
        self.n_views = n_views
        self.n_query_views = n_query_views
        self.image_size = image_size
        self.split = split
        self.transform = transform
        
        if cfg.data.num_samples == 'whole_data':
            # self.metadata = os.path.join(self.data_path, self.split+'_samples.txt')
            self.metadata = os.path.join('./data/data_split', self.split+'_samples.txt')
        else:
            self.metadata = os.path.join('./data/data_split', self.split+'_samples_%s.txt'%cfg.data.num_samples)
            # Just for quick experiments using only 256 samples for validation
            if self.split == 'val':
                num_val_samples = {'256': 'val_samples_256.txt',
                               '1k': 'val_samples_256.txt',
                               '10k': 'val_samples_1k.txt',
                               '50k': 'val_samples_10k.txt',
                               '100k': 'val_samples_10k.txt',
                               'whole_data': 'val_samples.txt'}
                self.metadata = os.path.join('./data/data_split', num_val_samples[cfg.data.num_samples])

        with open(self.metadata, 'r') as f:
            dir_list = f.readlines()
            self.dirs = [d.strip() for d in dir_list]
            
    def __len__(self):
        return len(self.dirs)
    
    # @staticmethod
    def load_image(self, img_path, size=[256, 256], mask=None):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != 256 or img.shape[1] != 256:
            img = cv2.resize(img, (size[0], size[1]), interpolation=cv2.INTER_AREA)
        if self.cfg.data.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=-1)
        if self.cfg.data.grayscale_3ch:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.stack([img, img, img], axis=-1)
        if self.cfg.data.white_background:
            img[mask==0] = 255.0
        return img
    
    def load_metadata(self, metadata_path):
        metadata = np.load(metadata_path, allow_pickle=True)
        # pose params is a dictionary with the following keys:
        # "elevation", "azimuth", "roll", "distance"
        pose_params = metadata['pose_params'].item()

        return pose_params
    
    def get_camera_extrinsics(self, pose_params):
        # Convert elevation, azimuth, roll to rotation matrix and translation vector
        # elevation: angle in degrees
        # azimuth: angle in degrees
        # roll: angle in degrees
        # distance: distance from the origin
        elevation = pose_params['elevation']
        azimuth = pose_params['azimuth']
        roll = pose_params['roll']
        distance = pose_params['distance']
        
        R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)
        roll_rad = torch.tensor([[0, 0, roll]]) * (torch.pi / 180.0)
        R_roll = euler_angles_to_matrix(roll_rad, "XYZ")
        R_final = torch.bmm(R, R_roll) # (1, 3, 3)
        T_final = T # (1, 3)
        return R_final.squeeze(0), T_final.squeeze(0)
        

    def __getitem__(self, idx):
        # Directory for a protein sample
        self.protein_path = os.path.join(self.data_path, self.dirs[idx])
        protein_name = self.dirs[idx].split('/')[-1]
        
        # Select the input views
        self.input_views_idx = np.random.choice(np.arange(25), size=self.n_views, replace=False)
        self.query_view_idx = np.random.choice(np.array(list(set(np.arange(25)) - set(self.input_views_idx))), size=self.n_query_views, replace=False)
        self.input_views = []
        for v in self.input_views_idx:
            img_path = os.path.join(self.protein_path, f"depth_map_with_tip_convolution_256_{v}.png")
            if self.cfg.data.white_background:
                mask = np.load(os.path.join(self.protein_path, f"depth_map_{v}.npz"))["mask_upsampled_256"]
            else:
                mask = None 
            img = torch.tensor(self.load_image(img_path, self.image_size, mask).astype(np.float32)/255.0) # normalize to [0, 1]
            img = torch.permute(img, (2, 0, 1)).contiguous()
            
            if self.cfg.data.use_depth:
                depth_path = os.path.join(self.protein_path, f"depth_map_{v}.npz")
                depth = np.load(depth_path)["depth_map_with_tip_convolution_256"] # (256, 256)
                depth = depth / np.max(depth) # normalize depth to [0, 1] by dividing by max depth 
                depth = np.expand_dims(depth, axis=0) # (1, 256, 256)
                # convert to tensor
                depth = torch.tensor(depth, dtype=torch.float32)
                img = torch.cat([img, depth], dim=0) # [(3, 256, 256), (1, 256, 256)] --> (4, 256, 256)
            
            self.input_views.append(img)
        self.input_views = torch.stack(self.input_views, dim=0) # (n_views, 3, 256, 256)

        # self.input_cameras = self.create_cameras_with_identity_extrinsics(self.n_views, self.focal_length, self.principal_point, self.image_size)

        self.query_views = []
        for v in self.query_view_idx:
            img_path = os.path.join(self.protein_path, f"depth_map_with_tip_convolution_256_{v}.png")
            if self.cfg.data.white_background:
                mask = np.load(os.path.join(self.protein_path, f"depth_map_{v}.npz"))["mask_upsampled_256"]
            else:
                mask = None
            img = torch.tensor(self.load_image(img_path, self.image_size, mask).astype(np.float32)/255.0) # normalize to [0, 1]
            img = torch.permute(img, (2, 0, 1)).contiguous()
            # img = img * 2.0 - 1.0 # (3, 256, 256)
            self.query_views.append(img)
        self.query_views = torch.stack(self.query_views, dim=0)

        # get the rotation matrix and translation vector for the query view
        self.metadata_path = os.path.join(self.protein_path, f"depth_map_{self.query_view_idx[0]}.npz")
        pose_params = self.load_metadata(self.metadata_path)
        self.R, self.T = self.get_camera_extrinsics(pose_params)
        
        return self.input_views, self.query_views, self.R, self.T 

class AutoEncoderDataset(Dataset):
    def __init__(self, cfg, data_path, image_size=[256, 256], split='train', transform=None):
        self.cfg = cfg
        self.data_path = data_path
        self.image_size = image_size
        self.split = split
        self.transform = transform
        
        if cfg.data.num_samples == 'whole_data':
            # self.metadata = os.path.join(self.data_path, self.split+'_samples.txt')
            self.metadata = os.path.join('./data/data_split', self.split+'_samples.txt')
        else:
            self.metadata = os.path.join('./data/data_split', self.split+'_samples_%s.txt'%cfg.data.num_samples)
            # Just for quick experiments using only 256 samples for validation
            if self.split == 'val':
                num_val_samples = {'256': 'val_samples_256.txt',
                               '1k': 'val_samples_256.txt',
                               '10k': 'val_samples_1k.txt',
                               '50k': 'val_samples_10k.txt',
                               '100k': 'val_samples_10k.txt',
                               'whole_data': 'val_samples.txt'}
                self.metadata = os.path.join('./data/data_split', num_val_samples[cfg.data.num_samples])

        with open(self.metadata, 'r') as f:
            dir_list = f.readlines()
            self.dirs = [d.strip() for d in dir_list]
            
    def __len__(self):
        return len(self.dirs)
    
    # @staticmethod
    def load_image(self, img_path, size=[256, 256]):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != 256 or img.shape[1] != 256:
            img = cv2.resize(img, (size[0], size[1]), interpolation=cv2.INTER_AREA)
        if self.cfg.data.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=-1)
        if self.cfg.data.grayscale_3ch:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.stack([img, img, img], axis=-1)
        return img
    
    
    def __getitem__(self, idx):
        # Directory for a protein sample
        self.protein_path = os.path.join(self.data_path, self.dirs[idx])
        
        view_idx = np.random.choice(np.arange(25), size=1, replace=False)
        img_path = os.path.join(self.protein_path, f"depth_map_with_tip_convolution_256_{view_idx[0]}.png")
        img = torch.tensor(self.load_image(img_path, self.image_size).astype(np.float32)/255.0) # normalize to [0, 1]
        img = torch.permute(img, (2, 0, 1)).contiguous()
        if self.cfg.data.use_depth:
            depth_path = os.path.join(self.protein_path, f"depth_map_{view_idx[0]}.npz")
            depth = np.load(depth_path)["depth_map_with_tip_convolution_256"] # (256, 256)
            if not self.cfg.data.use_raw_depth:
                depth = depth / np.max(depth) # normalize depth to [0, 1] by dividing by max depth 
            depth = np.expand_dims(depth, axis=0) # (1, 256, 256)
            # convert to tensor
            depth = torch.tensor(depth, dtype=torch.float32)
            img = torch.cat([img, depth], dim=0) # [(3, 256, 256), (1, 256, 256)] --> (4, 256, 256)
        # img = img * 2.0 - 1.0  # (3, 256, 256)
        # img = img.unsqueeze(0) # add batch size dimenstion (1, 3, 256, 256)
        return img