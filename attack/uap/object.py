import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

from utils.convertor import FormatConverter



class PatchManager:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.patch = None
    
    def init_diffusion(self, diffusion):
        self.diffusion = diffusion

    def init_delta(self):
        self.delta = torch.zeros_like(self.latent_init_ori)

    def init(self, patch_file=None):
        init_mode = self.cfg.INIT
        if patch_file is None:
            if init_mode == 'diffusion':
                self.latent, self.latent_init, self.latent_init_fft, self.all_latents, self.noise_pred_list, self.uncond_embeddings_list = self.diffusion.generate_latent()
                self.latent_init_ori = self.latent_init.clone().detach()

                self.start_step=0
                self.noise_pred_list = self.noise_pred_list[self.start_step:]
                #self.latent_init = self.all_latents[self.start_step].detach() #spatial domain

                self.latent_init_fft = torch.fft.fftn(self.all_latents[self.start_step])
                self.latent_init_fft_ori = self.latent_init_fft.clone().detach()
                self.latent_init_fft.requires_grad_(True)
                self.latent_init = torch.fft.ifftn(self.latent_init_fft).real

                #self.latent_init.requires_grad_(True) #spatial domain
                self.uncond_embedding_ori = self.uncond_embeddings_list[self.start_step].clone().detach()
                self.uncond_embeddings_list[self.start_step].requires_grad_(True)

                #self.latent = self.diffusion.update_latent_with_noise(self.latent_init,self.noise_pred_list,start_step = self.start_step)
                self.latent = self.diffusion.update_latent_with_noise_mask(self.latent_init,self.uncond_embeddings_list,self.noise_pred_list,start_step = self.start_step)
                self.patch = self.diffusion.latent2img(self.latent).to(self.device)
                self.patch_ori = self.patch.clone().detach()

            else:
                self.generate(init_mode)
        else:
            self.read(patch_file)
        if init_mode !='diffusion':
            self.patch.requires_grad = True

    def read(self, patch_file):
        print('Reading patch from file: ' + patch_file)
        if patch_file.endswith('.pth'):
            patch = torch.load(patch_file, map_location=self.device)
            # patch.new_tensor(patch)
            print(patch.shape, patch.requires_grad, patch.is_leaf)
        else:
            patch = Image.open(patch_file).convert('RGB')
            patch = FormatConverter.PIL2tensor(patch)
        if patch.ndim == 3:
            patch = patch.unsqueeze(0)
        self.patch = patch.to(self.device)

    def generate(self, init_mode='random'):
        height = self.cfg.HEIGHT
        width = self.cfg.WIDTH
        if init_mode.lower() == 'random':
            print('Random initializing a universal patch')
            patch = torch.rand((1, 3, height, width))
        elif init_mode.lower() == 'gray':
            print('Gray initializing a universal patch')
            patch = torch.full((1, 3, height, width), 0.5)
        elif init_mode.lower() == 'white':
            print('White initializing a universal patch')
            patch = torch.full((1, 3, height, width), 1.0)
        else:
            assert False, "Patch initialization mode doesn't exist!"
        self.patch = patch.to(self.device)

    def total_variation(self):
        adv_patch = self.patch[0]
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), 0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), 0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
        tv = tvcomp1 + tvcomp2
        return tv / torch.numel(adv_patch)
    

    def update_(self, patch_new):
        del self.patch
        self.patch = patch_new.detach()
        self.patch.requires_grad = True
    
    def update_latent_(self,latent_init_new):
        del self.patch
        del self.latent
        del self.latent_init
        latent_init = torch.fft.ifftn(latent_init_new).real 
        #latent_init = latent_init_new.detach() # spatial domain
        latent, latent_init, noise_pred_list = self.diffusion.update_latent(latent_init,uncond_embeddings_list=self.uncond_embeddings_list,start_step=self.start_step)
        
        self.latent_init_fft = latent_init_new.detach()
        self.latent_init_fft.requires_grad_(True)
        self.uncond_embeddings_list[self.start_step].requires_grad_(True)
        self.latent_init = torch.fft.ifftn(self.latent_init_fft).real
        
        #self.latent_init = latent_init.detach() # spatial domain
        #self.latent_init.requires_grad_(True)   # spatial domain
        self.noise_pred_list = noise_pred_list
        #self.latent = self.diffusion.update_latent_with_noise(self.latent_init,self.noise_pred_list,start_step=self.start_step)
        self.latent = self.diffusion.update_latent_with_noise_mask(self.latent_init,self.uncond_embeddings_list, self.noise_pred_list,start_step=self.start_step)
        self.patch = self.diffusion.latent2img(self.latent)

    @torch.no_grad()
    def clamp_(self, p_min=0, p_max=1):
        torch.clamp_(self.patch, min=p_min, max=p_max)
    
    