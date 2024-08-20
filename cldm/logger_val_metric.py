import os
import time
from datetime import datetime

import numpy as np
from pytorch_lightning import LightningModule, Trainer
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

class LoggerValMetric(Callback):
    def __init__(self, batch_frequency=1, max_images=16, split='val',clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.split = split
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.tensor2img = torchvision.transforms.ToPILImage()

    def log_local(self, save_dir, split, images, pl_module, batch_idx):
        root = os.path.join(save_dir, "validation_res", split)
        os.makedirs(root, exist_ok=True)
        
        ct = images['reconstruction']
        output_key = list(filter(lambda x: 'samples' in x, images.keys()))[0]
        # print(images.keys(), output_key)
        output = images[output_key]
        
        if self.rescale:
            ct = (ct + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            output = (output + 1.0) / 2.0
        
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0, return_full_image=True)
        ssim_val, ssim_img = ssim(output, ct)
        # print(ssim_val.item())
        pl_module.log('val/metric', ssim_val.item(), on_step=True, on_epoch=True, batch_size=ct.shape[0])
        
        psnr = PeakSignalNoiseRatio()
        psnr_val = psnr(output, ct)
        pl_module.log('val/psnr', psnr_val.item(), on_step=True, on_epoch=True, batch_size=ct.shape[0])
        
        if pl_module.current_epoch % 1 == 0:
            for i in range(ct.shape[0]):
                self.tensor2img(ssim_img[i]).save(os.path.join(root,f'ep{pl_module.current_epoch:04}_step{pl_module.global_step:04}_bi{batch_idx:05}_ssim{ssim_val:04}_psnr{psnr_val:04}_{i}.png'))
        
                
    def log_val(self, pl_module, batch, batch_idx, split="val"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, N=12, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_val(pl_module, batch, batch_idx, split=self.split)
    