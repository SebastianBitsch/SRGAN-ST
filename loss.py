from os import listdir
from random import choices
from torch import nn, Tensor, linalg
import torch

import torch
from skimage import io, img_as_float
# from torchvision import io
import numpy as np

class EuclidLoss(nn.Module):
    """"""

    def __init__(self) -> None:
        super(EuclidLoss, self).__init__()
        self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor) -> Tensor:
        # Input: torch.Size([16, 3, 96, 96])
        
        # torch.Size([16*3*96*96])
        sr_tensor = torch.flatten(sr_tensor)
        gt_tensor = torch.flatten(gt_tensor)
        
        loss = self.pdist(sr_tensor, gt_tensor)

        return loss


class TextureLoss(nn.Module):
    """ As described in Eq.2 in GramGAN - but without candidate patches"""

    def __init__(self, ord = 2) -> None:
        super(TextureLoss, self).__init__()
        self.ord = ord

    def gram_matrix(self, input):
        """ from: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html"""
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)
    
    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor) -> Tensor:
        
        G_e = self.gram_matrix(sr_tensor)
        G_i = self.gram_matrix(gt_tensor)

        loss = linalg.matrix_norm(G_e - G_i, ord=self.ord) ** 2
        
        return loss

class PatchWiseTextureLoss(nn.Module):
    """ As described in Eq.2 in GramGAN """

    def __init__(self, device: str, alpha: float = 1.0, beta: float = 1.0, ord = 2, k = 50, batch_size = 16) -> None:
        super(PatchWiseTextureLoss, self).__init__()
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.ord = ord
        self.k = k
        self.batch_size = batch_size

        # See https://pytorch.org/functorch/stable/generated/functorch.vmap.html
        self.batched_gram_matrix = torch.vmap(self.gram_matrix)

    def gram_matrix(self, input: Tensor) -> Tensor:
        """ from: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html"""
        b, c, d = input.size()
        features = input.view(b, c * d)
        G = torch.mm(features, features.t())

        return G.div(b * c * d)
    
    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor) -> Tensor:
        
        # In: torch.Size([16, 3, 96, 96]), Out: torch.Size([16, 3, 3])
        G_e = self.batched_gram_matrix(sr_tensor)
        G_i = self.batched_gram_matrix(gt_tensor)

        p_dists = []
        patches = [] # 50 x 3 x 96 x 96
        
        fpath = "data/ImageNet/SRGAN/train/"
        for fname in choices(listdir(fpath), k = self.k):
            # patch = io.read_image(fpath + fname)
            patch = img_as_float(io.imread(fpath + fname))
            patch = torch.tensor(patch, requires_grad = True, device=self.device)
            patch = torch.permute(patch, (-1, 0, 1))
            
            G_p = self.gram_matrix(patch) # 3 x 3
            dist = (self.alpha * linalg.matrix_norm(G_p - G_i, ord=self.ord) ** 2) + (self.beta * linalg.matrix_norm(G_p - G_e, ord=self.ord) ** 2)

            patches.append(patch)
            p_dists.append(dist)

        
        p_star_i = torch.argmin(torch.cat(p_dists, dim=0).reshape(self.k, self.batch_size), dim=0) # 1 x 16
        
        patches = torch.stack(patches)
        p_star = patches[p_star_i]
    
        G_pi = self.batched_gram_matrix(p_star) # 16 x 3 x 3

        L = linalg.matrix_norm(G_e - G_pi, ord=1) # See Eq. 4 GramGAN

        return torch.mean(L)


def gram_matrix(input):
    """ from: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html"""
    b, c, d = input.size()
    features = input.view(b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    return G.div(b * c * d)