import random
import torch
from torch import nn
import torch.nn.functional as F
import clip

MEAN=[0.48145466, 0.4578275, 0.40821073] 
STD=[0.26862954, 0.26130258, 0.27577711]
INPUT_SIZE = {'RN50': 224,
 'RN101': 224,
 'RN50x4': 288,
 'RN50x16': 384,
 'RN50x64': 448,
 'ViT-B/32': 224,
 'ViT-B/16': 224,
 'ViT-L/14': 224,
 'ViT-L/14@336px': 336}
#MODELS_LIST = ['RN50','RN101','ViT-B/32','ViT-B/16','ViT-L/14']
#MODELS_LIST = ['RN50','RN50x4','RN50x16','RN50x64','ViT-L/14@336px']
#MODELS_LIST = ['ViT-L/14']
MODELS_LIST = ['ViT-B/32','RN50']
class ClipModel:
    def __init__(self, device, models_list = MODELS_LIST):
        self.device = device
        self.models_list= models_list
        self.models = {model_name: clip.load(model_name,device = self.device)[0].requires_grad_(False) for model_name in self.models_list}
        self.mean = torch.tensor(MEAN,device = self.device).view(1,3,1,1)
        self.std = torch.tensor(STD, device = self.device).view(1,3,1,1)
    def normalize(self, x):
        return (x - self.mean) / self.std
    def resize(self, x, size):
        return F.interpolate(x,(size, size))
    def encode_image(self,x,mode='random'):
        if mode == 'random':
            model_name = random.choice(self.models_list)
            model = self.models[model_name]
            size = INPUT_SIZE[model_name]
            x = self.resize(x, size)
            x = self.normalize(x)
            x = model.encode_image(x)
            return x, model_name
        if mode == 'full':
            output = {}
            for model_name in self.models_list:
                size = INPUT_SIZE[model_name]
                output[model_name] = self.models[model_name].encode_image(self.normalize(self.resize(x,size)))
            return output
        else:
            model = self.models[mode]
            size = INPUT_SIZE[mode]
            x = self.resize(x, size)
            x = self.normalize(x)
            x = model.encode_image(x)
            return x, mode
    
            
    def encode_text(self,text,mode='random'):
        token = clip.tokenize(text).to(self.device)
        if mode == 'random':
            model_name = random.choice(self.models_list)
            model = self.models[model_name]
            text = model.encode_text(token)
            return text, model_name
        if mode == 'full':
            output = {}
            for model_name in self.models_list:
                output[model_name] = self.models[model_name].encode_text(token)
            return output
        else:
            model = self.models[mode]
            text = model.encode_text(token)
            return x, mode
    def tokenize(self, prompt):
        self.text = clip.tokenize(prompt).to(self.device)
        self.text_emb = self.model.encode_text(self.text)
    def init_threshold(self,feature):
        sim = ((feature @ self.text_emb.t()) / (feature.norm(2) * self.text_emb.norm(2))).item()
        self.threshold = sim + 0.01
        
    
    def forward(self, x):
        x = self.encode_image(x)
        sim = x @ self.text_emb.t() / (x.norm(2) * self.text_emb.norm(2))
        sim = sim.clamp(max = self.threshold)
        return sim

