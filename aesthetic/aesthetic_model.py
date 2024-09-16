import torch
from torch import nn
import numpy as np

class AestheticMlp(nn.Module):
    def __init__(self, input_size, xcol="emb", ycol="avg_rating"):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)
   
    def get_aesthetic_score(self, image_features, is_batched=False):
        features = image_features.cpu().detach().numpy()
        order = 2
        axis = -1
        l2 = np.atleast_1d(np.linalg.norm(features, order, axis))
        l2[l2 == 0] = 1
        im_emb_arr = features / np.expand_dims(l2, axis)
        prediction = self.forward(
            torch.from_numpy(im_emb_arr)
            .to('cuda')
            .type(torch.cuda.FloatTensor)
        )
        if is_batched:
            return prediction[:, 0].tolist()
        else:
            return prediction.item()
def make_model(dim=768,device='cuda'):
    model = AestheticMlp(dim)
    s = torch.load("./aesthetic/sac+logos+ava1-l14-linearMSE.pth")
    model.load_state_dict(s)
    model.eval()
    model.to(device)
    return model