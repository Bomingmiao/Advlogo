import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import random
from diffusers import StableDiffusionPipeline, DDIMScheduler,AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, AutoTokenizer
from tqdm import tqdm


RES = 512
TRANSFORM = transforms.Compose([transforms.Resize((RES,RES)),
                               transforms.ToTensor()])


class Diffusion:
    def __init__(self, checkpoint_path,prompt,num_inference_steps,guidance_scale,seed,latent,device):
        self.device = device
        self.prompt = prompt
        self.seed = seed
        self.latent = latent
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.text_encoder = CLIPTextModel.from_pretrained(
    checkpoint_path,
    subfolder="text_encoder",
    revision=None,
).requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            checkpoint_path, subfolder="unet", revision = None
        ).requires_grad_(False)
        self.vae = AutoencoderKL.from_pretrained(
        checkpoint_path, subfolder="vae", revision=None
    ).requires_grad_(False)
        self.scheduler = DDIMScheduler.from_pretrained(checkpoint_path,subfolder="scheduler",revision = None)
        self.scheduler.set_timesteps(self.num_inference_steps)
        
    
    def generate_latent(self,seed=None):
        uncond_input = self.tokenizer(
        "",
        truncation=True,
        padding="max_length",
        max_length=self.tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(self.device)
        cond_input = self.tokenizer(
        self.prompt,
        truncation=True,
        padding="max_length",
        max_length=self.tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(self.device)
        self.text_encoder.to(self.device)
        self.uncond_embeddings = self.text_encoder(uncond_input)[0]
        self.cond_embeddings = self.text_encoder(cond_input)[0]
        height = width = RES//8
        if seed is not None:
            torch.manual_seed(seed)
            noise  = torch.randn(1, self.unet.in_channels, height, width).to(self.device)
        elif self.seed is not None:
            generator = torch.Generator().manual_seed(self.seed)
            noise = torch.randn((1,self.unet.in_channels,height,width),generator = generator).to(self.device)
        else:
            noise = torch.randn((1, self.unet.in_channels,height,width),device = self.device)
        if self.latent is None:
            latent = noise
        else:
            latent = self.latent.detach().to(self.device)
        latent_init = latent.clone().detach()
        latent_init_fft = torch.fft.fftn(latent_init)
        latent_init = torch.fft.ifftn(latent_init_fft).float()
        all_latents = [latent.detach()]
        self.unet.to(self.device)
        if self.guidance_scale != 1:
            context = torch.cat([self.uncond_embeddings,self.cond_embeddings])
            uncond_embeddings_list = [self.uncond_embeddings] * self.num_inference_steps
        else:
            context = self.cond_embeddings
        self.context = context
        noise_pred_list = []
        for i,t in tqdm(enumerate(self.scheduler.timesteps),total=len(self.scheduler.timesteps),desc="generating images with prompts"):
            if self.guidance_scale != 1:
                latent_input = torch.cat([latent]*2)

            else:
                latent_input = latent

            noise_pred = self.unet(latent_input,t,context)['sample']
            if self.guidance_scale != 1:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale* (noise_pred_cond - noise_pred_uncond)
            noise_pred_list.append(noise_pred.detach())
            latent = self.scheduler.step(noise_pred,t,latent)['prev_sample']
            all_latents.append(latent.detach())


        return latent, latent_init, latent_init_fft,all_latents, noise_pred_list, uncond_embeddings_list
    
    def latent2img(self, latent):
        latent_last = 1/self.vae.config.scaling_factor *latent
        img = self.vae.to(self.device).decode(latent_last)['sample']
        img = img / 2 + 0.5
        img = img.clamp(0,1)
        return img
        
    
    def update_latent(self,latent_new,uncond_embeddings_list=None,start_step=0):
        latent = latent_new
        noise_pred_list = []
        for i, t in tqdm(enumerate(self.scheduler.timesteps[start_step:]),total=len(self.scheduler.timesteps[start_step:]),desc="updating latents"):

            if self.guidance_scale != 1:

                latent_input = torch.cat([latent]*2)
                if uncond_embeddings_list is None:
                    context = self.context
                else:
                    context = torch.cat([uncond_embeddings_list[start_step+i],self.cond_embeddings])
            else:
                latent_input = latent
                context = self.cond_embeddings
            noise_pred = self.unet(latent_input,t,context)['sample']
            if self.guidance_scale != 1:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale*(noise_pred_cond - noise_pred_uncond)
            noise_pred_list.append(noise_pred.detach())
            latent = self.scheduler.step(noise_pred,t,latent)['prev_sample']
        return latent, latent_new, noise_pred_list
    
    def update_noise_pred(self,uncond_embeddings_list, all_latents, noise_pred_list, start_step):
        latent = all_latents[start_step]
        for i, _ in tqdm(enumerate(noise_pred_list[start_step:]),total = len(noise_pred_list[start_step:]),desc="updating noise predicted"):
            context = torch.cat([uncond_embeddings_list[i],self.cond_embeddings])
            latent_input = torch.cat([latent]*2)
            noise_pred = self.unet(latent_input,self.scheduler.timesteps[start_step+i],context)['sample']
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            latent = self.scheduler.step(noise_pred,self.scheduler.timesteps[start_step+i],latent)['prev_sample']
            noise_pred_list[start_step+i]=noise_pred.detach()
            all_latents[start_step+i+1]=latent.detach()
        return all_latents, noise_pred_list
    
    def get_latent_with_mask(self,uncond_embeddings, noise_pred_list, all_latents, start_step):
        latent = all_latents[start_step]
        context = torch.cat([uncond_embeddings,self.cond_embeddings])
        latent_input = torch.cat([latent]*2)
        noise_pred = self.unet(latent_input,self.scheduler.timesteps[start_step],context)['sample']
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latent = self.scheduler.step(noise_pred,self.scheduler.timesteps[start_step],latent)['prev_sample']
        if start_step + 1 == len(noise_pred_list):
            pass
        else:            
            for i, _ in tqdm(enumerate(noise_pred_list[start_step+1:]),total = len(noise_pred_list[start_step+1:]),desc="updating latents with predicted noise"):
                noise_pred = noise_pred_list[start_step+i+1]
                latent = self.scheduler.step(noise_pred,self.scheduler.timesteps[start_step+i+1],latent)['prev_sample']
        return latent
        
    def update_latent_with_noise(self,latent,noise_pred_list,start_step=0):
        for i, t in tqdm(enumerate(self.scheduler.timesteps[start_step:]),total=len(self.scheduler.timesteps[start_step:]),desc="updating latents with predicted noise"):
            noise_pred = noise_pred_list[i]
            latent = self.scheduler.step(noise_pred,t,latent)['prev_sample']
        return latent
    def update_latent_with_noise_mask(self,latent,uncond_embeddings_list,noise_pred_list,start_step=0):
        for i, t in tqdm(enumerate(self.scheduler.timesteps[start_step:]),total=len(self.scheduler.timesteps[start_step:]),desc="updating latents with predicted noise"):
            if i==0:
                uncond_embedding = uncond_embeddings_list[i]
                context = torch.cat([uncond_embedding,self.cond_embeddings])
                latent_input = torch.cat([latent]*2)
                noise_pred = self.unet(latent_input,self.scheduler.timesteps[start_step],context)['sample']
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latent = self.scheduler.step(noise_pred,self.scheduler.timesteps[start_step],latent)['prev_sample']
            else:
                noise_pred = noise_pred_list[i]
                latent = self.scheduler.step(noise_pred,t,latent)['prev_sample']
        return latent
    
    def read_patch(self, patch_path):
        patch = TRANSFORM(Image.open(patch_path)).unsqueeze(0)
        latent = self.vae.encode(patch).latent_dist.sample().to(self.device)
        latent = latent * self.vae.config.scaling_factor
        self.latent = latent
    
    
    
    
    
    def null_optimization(self,latent,uncond_embeddings,iter=10,epsilon=1e-5):
        self.unet.to(self.device)
        uncond_embeddings_list = []
        latent_cur = latent.clone().detach()
        for i, t in tqdm(enumerate(self.scheduler.timesteps),total = len(self.scheduler.timesteps),desc="optimizing null embeddings"):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = torch.optim.Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            with torch.no_grad():
                noise_pred_cond = self.unet(latent_cur,t,self.cond_embeddings)['sample']
                latent_prev_cond = self.scheduler.step(noise_pred_cond,t,latent_cur)['prev_sample']
            for j in tqdm(range(iter)):
                noise_pred_uncond = self.unet(latent_cur,t,uncond_embeddings)['sample']
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latent_prev = self.scheduler.step(noise_pred, t, latent_cur)['prev_sample']
                loss = F.mse_loss(latent_prev_cond, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                if loss_item < epsilon + i * 2e-5:
                    break
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, self.cond_embeddings])
                latent_input = torch.cat([latent]*2)
                noise_pred = self.unet(latent_input,t,context)['sample']
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latnet_cur = self.scheduler.step(noise_pred, t, latent_cur)['prev_sample']
        return uncond_embeddings_list, latent_cur


    def get_attention_maps(self,sample,timestep):
        self.unet.to(self.device)
        encoder_hidden_states = self.cond_embeddings
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.unet.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.unet.dtype)

        emb = self.unet.time_embedding(t_emb)


        sample = self.unet.conv_in(sample)

        # 3. down
        #down_block_res_samples = (sample,)
        down_block_samples = []

        for downsample_block in self.unet.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,

                )

            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)



            #down_block_res_samples += res_samples
            down_block_samples.append(sample)
            down_block_samples = down_block_samples[:1]
        return down_block_samples
    
    

    
    

        
    