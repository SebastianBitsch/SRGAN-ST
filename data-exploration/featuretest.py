import torch
from skimage import io 
from model import Discriminator
from loss import ContentLoss
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


device = torch.device('cuda')
image = io.imread("/zhome/45/8/156251/Desktop/SRGAN-ST-1/data/Set5/GTmod12/butterfly.png")
image = torch.tensor(image.astype('float32'), device = 'cuda')
image = image.permute(2, 0, 1)[:,:192,:192]
#image = image.to(device='cuda')
con = ContentLoss({"features.17":1/8}, torch.device('cuda'))
feat_image  = con.feature_extractor(image)["features.17"]
feature_map = feat_image.squeeze(0)
gray_scale = torch.sum(feature_map,0)
gray_scale = gray_scale / feature_map.shape[0]


#print(feature_map.shape)
feature_image = gray_scale.cpu().numpy()  # convert tensor to numpy array




dis = Discriminator().to(device)
#image = image.to(device = 'cuda')
print(image.unsqueeze(0).shape)
out = dis.features[0](image.unsqueeze(0))
#out = dis.features[5].weight
#out = out.permute(3,2,0,1)
# Flatten the first two dimensions
print(out.shape)
# flattened_t = out.reshape(-1,64,64)
# print(flattened_t.shape)
d_grayscale = out.sum(0)
d_grayscale = d_grayscale.sum(0)
print(d_grayscale.shape)
d_feature_image = d_grayscale.cpu().detach().numpy()  # convert tensor to numpy array
# save the feature_image numpy array
np.save('data/features/D_feature_image.npy', d_feature_image)

