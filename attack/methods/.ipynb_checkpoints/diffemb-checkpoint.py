import copy
import sys

from abc import ABC, abstractmethod
import torch
from torch.optim.optimizer import Optimizer
import torch.distributed as dist
import numpy as np

num_iter = 0
update_pre = 0
# patch_tmp = None

import sys
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.append(PROJECT_ROOT)
from utils import FormatConverter
import pickle

class DiffEmbAttack(Optimizer):
    """An Attack Base Class"""

    def __init__(self, loss_func, norm: str, cfg, device: torch.device, detector_attacker):
        """

        :param loss_func:
        :param norm: str, [L0, L1, L2, L_infty]
        :param cfg:
        :param detector_attacker: this attacker should have attributes vlogger

        Args:
            loss_func ([torch.nn.Loss]): [a loss function to calculate the loss between the inputs and expeced outputs]
            norm (str, optional): [the attack norm and the choices are [L0, L1, L2, L_infty]]. Defaults to 'L_infty'.
            epsilons (float, optional): [the upper bound of perturbation]. Defaults to 0.05.
            max_iters (int, optional): [the maximum iteration number]. Defaults to 10.
            step_lr (float, optional): [the step size of attack]. Defaults to 0.01.
            device ([type], optional): ['cpu' or 'cuda']. Defaults to None.
        """
        defaults = dict(lr=cfg.STEP_LR)
        params = [detector_attacker.patch_obj.uncond_embeddings_list[detector_attacker.patch_obj.start_step]]
        super().__init__(params, defaults)

        self.loss_fn = loss_func
        self.cfg = cfg
        self.detector_attacker = detector_attacker
        self.device = device
        self.norm = norm
        self.max_epsilon = cfg.EPSILON
        self.max_iters = cfg.MAX_EPOCH
        self.iter_step = cfg.ITER_STEP
        self.attack_class = cfg.ATTACK_CLASS
        self.step_lr = cfg.STEP_LR
        self.optimizer = torch.optim.AdamW([detector_attacker.patch_obj.uncond_embeddings_list[detector_attacker.patch_obj.start_step]],lr=self.step_lr)


    def logger(self, detector, adv_tensor_batch, bboxes, loss_dict):
        vlogger = self.detector_attacker.vlogger
        # TODO: this is a manually appointed logger iter num 77(for INRIA Train)
        if vlogger:
            # print(loss_dict['loss'], loss_dict['det_loss'], loss_dict['tv_loss'])
            vlogger.note_loss(loss_dict['loss'], loss_dict['det_loss'], loss_dict['rec_loss'])
            if vlogger.iter % 77 == 0:
                filter_box = self.detector_attacker.filter_bbox
                vlogger.write_tensor(self.detector_attacker.universal_patch[0], 'adv patch')
                plotted = self.detector_attacker.plot_boxes(adv_tensor_batch[0], filter_box(bboxes[0]))
                vlogger.write_cv2(plotted, f'{detector.name}')

    def non_targeted_attack(self, ori_tensor_batch, detector):
        losses = []
        losses_det = []
        for iter in range(self.iter_step):
            if iter > 0: ori_tensor_batch = ori_tensor_batch.clone()
            adv_tensor_batch = self.detector_attacker.uap_apply(ori_tensor_batch)
            adv_tensor_batch = adv_tensor_batch.to(detector.device)
            # detect adv img batch to get bbox and obj confs
            bboxes, confs, cls_array = detector(adv_tensor_batch).values()

            if hasattr(self.cfg, 'class_specify'):
                # TODO: only support filtering a single cls now
                attack_cls = int(self.cfg.ATTACK_CLASS)
                confs = torch.cat(
                    ([conf[cls == attack_cls].max(dim=-1, keepdim=True)[0] for conf, cls in zip(confs, cls_array)]))
            elif hasattr(self.cfg, 'topx_conf'):
                # attack top x confidence
                # print(confs.size())
                confs = torch.sort(confs, dim=-1, descending=True)[0][:, :self.cfg.topx_conf]
                confs = torch.mean(confs, dim=-1)
            else:
                # only attack the max confidence
                confs = confs.max(dim=-1, keepdim=True)[0]

            detector.zero_grad()
            # print('confs', confs)
            loss_dict = self.attack_loss(confs)
            loss = loss_dict['loss']
            det_loss = loss_dict['det_loss']
            # print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            # print(self.detector_attacker.patch_obj.latent.grad)
            losses.append(float(loss))
            losses_det.append(float(det_loss.detach()))
            
            # update patch. for optimizer, using optimizer.step(). for PGD or others, using clamp and SGD.
            self.patch_update()
        # print(adv_tensor_batch, bboxes, loss_dict)
        # update training statistics on tensorboard
        self.logger(detector, adv_tensor_batch, bboxes, loss_dict)
        return torch.tensor(losses).mean(), torch.tensor(losses_det).mean()
    
    def patch_update(self, **kwargs):
        grad = (self.patch_obj.uncond_embeddings_list[self.patch_obj.start_step].grad)
        print(grad)
        torch.save(grad,'grad.pt')
        self.optimizer.step()
        with open('uncond_embeddings_list.pkl','wb') as f:
            pickle.dump(self.patch_obj.uncond_embeddings_list,f)
        self.patch_obj.uncond_embeddings_list[self.patch_obj.start_step].requires_grad=False
        self.patch_obj.update_noise_()
        self.optimizer = torch.optim.AdamW([self.patch_obj.uncond_embeddings_list[self.patch_obj.start_step]],lr=self.step_lr)
        patch_tmp = self.patch_obj.patch
        return patch_tmp
    
    
    def attack_loss(self,confs):
        obj_loss = self.loss_fn(confs = confs)
        #rec_loss = self.cfg.tv_eta * torch.nn.functional.mse_loss(self.detector_attacker.patch_obj.latent_ori, self.detector_attacker.patch_obj.latent)
        #rec_loss = self.cfg.tv_eta * torch.nn.functional.mse_loss(self.detector_attacker.patch_obj.patch_ori, self.detector_attacker.patch_obj.patch)
        #self.detector_attacker.patch_obj.attention_maps = self.detector_attacker.patch_obj.diffusion.get_attention_maps(self.detector_attacker.patch_obj.latent,0)
        #attn_loss = self.cfg.tv_eta * self.detector_attacker.patch_obj.get_attention_loss()
        
        self.detector_attacker.patch_obj.get_feature()
        sim_imgs = self.detector_attacker.patch_obj.get_imgs_similarity()
        #aes_score = self.detector_attacker.patch_obj.get_aesthetic_score()
        #sim_img_text = self.detector_attacker.patch_obj.get_img_text_similarity()
        rec_loss = self.cfg.tv_eta * (1-sim_imgs)
        #loss = obj_loss + rec_loss.to(obj_loss.device)
        loss = obj_loss
        out = {'loss': loss, 'det_loss':obj_loss, 'rec_loss': rec_loss}
        print(out)
        return out

    def parallel_non_targeted_attack(self, ori_tensor_batch, detector_attacker, detector):
        adv_tensor_batch, patch_tmp = detector_attacker.uap_apply(ori_tensor_batch)
        loss = []
        for iter in range(self.iter_step):
            preds, confs = detector(adv_tensor_batch)
            disappear_loss = self.attack_loss(confs,patch_tmp, self.patch_ori)
            loss.append(float(disappear_loss))
            detector.zero_grad()
            disappear_loss.backward()
            self.patch_update()
            adv_tensor_batch, _ = detector_attacker.uap_apply(ori_tensor_batch, universal_patch=patch_tmp)
       

        return patch_tmp
            

    @property
    def patch_obj(self):
        return self.detector_attacker.patch_obj

    

    def begin_attack(self):
        """
        to tell attackers: now, i'm begin attacking!
        """
        pass

    def end_attack(self):
        """
        to tell attackers: now, i'm stop attacking!
        """
        pass


            