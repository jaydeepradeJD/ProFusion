import os
import cv2
import numpy as np
import lpips    
import skimage.metrics
import torch


def setup_lpips(device=None):
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    return loss_fn_vgg

def get_lpips(loss_fn_vgg, gt, pred, device=None):
    '''
    Return LPIPS function
    '''
    # gt = (gt/255.0).astype(np.float32)
    # pred = (pred/255.0).astype(np.float32)
    gt_ = torch.from_numpy(gt).permute(2,0,1).unsqueeze(0)*2 - 1.0
    pred_ = torch.from_numpy(pred).permute(2,0,1).unsqueeze(0)*2 - 1.0
    if device is not None:
        lp = loss_fn_vgg(gt_.to(device), pred_.to(device)).detach().cpu().numpy().item()
    else:
        lp = loss_fn_vgg(gt_, pred_).detach().numpy().item()
    return lp


# gt_image_path = '/work/mech-ai-scratch/jrrade/Protein/upfusion/samples_for_metric_calculation/AF-A2RD68-F1-model_v4.pdb/AF-A2RD68-F1-model_v4.pdb_0.png'
# pred_image_1_path = '/work/mech-ai-scratch/jrrade/Protein/upfusion/samples_for_metric_calculation/AF-A2RD68-F1-model_v4.pdb/1/0.png'
# pred_image_3_path = '/work/mech-ai-scratch/jrrade/Protein/upfusion/samples_for_metric_calculation/AF-A2RD68-F1-model_v4.pdb/3/25.png'
# pred_image_6_path = '/work/mech-ai-scratch/jrrade/Protein/upfusion/samples_for_metric_calculation/AF-A2RD68-F1-model_v4.pdb/6/3.png'

# gt_image = cv2.imread(gt_image_path)
# gt_image[np.all(gt_image == [0, 0, 0], axis=-1)] = [255, 255, 255]
# gt_image = cv2.resize(gt_image, (256, 256))

# pred_image_1 = cv2.imread(pred_image_1_path)
# pred_image_3 = cv2.imread(pred_image_3_path)
# pred_image_6 = cv2.imread(pred_image_6_path)

# gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
# pred_image_1 = cv2.cvtColor(pred_image_1, cv2.COLOR_BGR2RGB)
# pred_image_3 = cv2.cvtColor(pred_image_3, cv2.COLOR_BGR2RGB)
# pred_image_6 = cv2.cvtColor(pred_image_6, cv2.COLOR_BGR2RGB)
# gt_image = (gt_image/255.0).astype(np.float32)
# pred_image_1 = (pred_image_1/255.0).astype(np.float32)
# pred_image_3 = (pred_image_3/255.0).astype(np.float32)
# pred_image_6 = (pred_image_6/255.0).astype(np.float32)

# #ssim
# ssim_1 = skimage.metrics.structural_similarity(gt_image, pred_image_1, channel_axis=-1, data_range=1)
# ssim_3 = skimage.metrics.structural_similarity(gt_image, pred_image_3, channel_axis=-1, data_range=1)
# ssim_6 = skimage.metrics.structural_similarity(gt_image, pred_image_6, channel_axis=-1, data_range=1)
# print('SSIM: ', ssim_1, ssim_3, ssim_6)

# #psnr
# psnr_1 = skimage.metrics.peak_signal_noise_ratio(gt_image, pred_image_1, data_range=1)
# psnr_3 = skimage.metrics.peak_signal_noise_ratio(gt_image, pred_image_3, data_range=1)
# psnr_6 = skimage.metrics.peak_signal_noise_ratio(gt_image, pred_image_6, data_range=1)
# print('PSNR: ', psnr_1, psnr_3, psnr_6)


# lpips_1 = get_lpips(gt_image, pred_image_1)
# lpips_3 = get_lpips(gt_image, pred_image_3)
# lpips_6 = get_lpips(gt_image, pred_image_6)
# print('LPIPS: ', lpips_1, lpips_3, lpips_6)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
directory = '/work/hdd/bcyr/jd23697/ProFusion/training_logs_diffusion_finetune/predict_diffusion_whole_data_diffusion_pretrained_sdv15_sd_locked_grayscale_3ch_afm_rays_ddim_1_30'
num_samples = 1000
SSIMs = []
PSNRs = []
LPIPSs = []
loss_fn_vgg = setup_lpips(device)
for i in range(num_samples):
    gt_image_path = os.path.join(directory, '%04d_gt.png'%(i))
    pred_image_path = os.path.join(directory, '%04d_pred.png'%(i))
    gt_image = cv2.imread(gt_image_path)
    pred_image = cv2.imread(pred_image_path)
    gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
    pred_image = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)
    gt_image = (gt_image/255.0).astype(np.float32)
    pred_image = (pred_image/255.0).astype(np.float32)
    ssim = skimage.metrics.structural_similarity(gt_image, pred_image, channel_axis=-1, data_range=1)
    psnr = skimage.metrics.peak_signal_noise_ratio(gt_image, pred_image, data_range=1)
    lpips_ = get_lpips(loss_fn_vgg, gt_image, pred_image, device)
    SSIMs.append(ssim)
    PSNRs.append(psnr)
    LPIPSs.append(lpips_)
    
print('SSIM: ', np.min(SSIMs), np.max(SSIMs), np.mean(SSIMs))
print('PSNR: ', np.min(PSNRs), np.max(PSNRs), np.mean(PSNRs))
print('LPIPS: ', np.min(LPIPSs), np.max(LPIPSs), np.mean(LPIPSs))