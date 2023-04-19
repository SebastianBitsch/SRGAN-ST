from os import listdir
from random import choices
from torch import nn, Tensor, linalg
import torch

import torch.nn.functional as F
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


class BBLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, ksize=3, pad=0, stride=3, dist_norm='l2', criterion='l1'):
        super(BBLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ksize = ksize
        self.pad = pad
        self.stride = stride
        self.dist_norm = dist_norm

        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss(reduction='mean')
        elif criterion == 'l2':
            self.criterion = torch.nn.L2loss(reduction='mean')
        else:
            raise NotImplementedError('%s criterion has not been supported.' % criterion)

    def batch_pairwise_distance(self, x, y=None):
        '''
        Input: x is a BxNxd matrix
               y is an optional BxMxd matirx
        Output: dist is a BxNxM matrix where dist[b,i,j] is the square norm between x[b,i,:] and y[b,j,:]
                if y is not given then use 'y=x'.
        i.e. dist[b,i,j] = ||x[b,i,:]-y[b,j,:]||^2
        '''
        B, N, d = x.size()
        if self.dist_norm == 'l1':
            x_norm = x.view(B, N, 1, d)
            if y is not None:
                y_norm = y.view(B, 1, -1, d)
            else:
                y_norm = x.view(B, 1, -1, d)
            dist = torch.abs(x_norm - y_norm).sum(dim=3)
        elif self.dist_norm == 'l2':
            x_norm = (x ** 2).sum(dim=2).view(B, N, 1)
            if y is not None:
                M = y.size(1)
                y_t = torch.transpose(y, 1, 2)
                y_norm = (y ** 2).sum(dim=2).view(B, 1, M)
            else:
                y_t = torch.transpose(x, 1, 2)
                y_norm = x_norm.view(B, 1, N)

            dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
            # Ensure diagonal is zero if x=y
            if y is None:
                dist = dist - torch.diag_embed(torch.diagonal(dist, dim1=-2, dim2=-1), dim1=-2, dim2=-1)
            dist = torch.clamp(dist, 0.0, np.inf)
            # dist = torch.sqrt(torch.clamp(dist, 0.0, np.inf) / d)
        else:
            raise NotImplementedError('%s norm has not been supported.' % self.dist_norm)

        return dist

    def forward(self, x, gt):
        p1 = F.unfold(x, kernel_size=self.ksize, padding=self.pad, stride=self.stride)
        B, C, H = p1.size()
        p1 = p1.permute(0, 2, 1).contiguous() # [B, H, C]

        p2 = F.unfold(gt, kernel_size=self.ksize, padding=self.pad, stride=self.stride)
        p2 = p2.permute(0, 2, 1).contiguous() # [B, H, C]

        gt_2 = F.interpolate(gt, scale_factor=0.5, mode='bicubic', align_corners = False)
        p2_2 = F.unfold(gt_2, kernel_size=self.ksize, padding=self.pad, stride=self.stride)
        p2_2 = p2_2.permute(0, 2, 1).contiguous() # [B, H, C]

        gt_4 = F.interpolate(gt, scale_factor=0.25, mode='bicubic',align_corners = False)
        p2_4 = F.unfold(gt_4, kernel_size=self.ksize, padding=self.pad, stride=self.stride)
        p2_4 = p2_4.permute(0, 2, 1).contiguous() # [B, H, C]
        p2_cat = torch.cat([p2, p2_2, p2_4], 1)

        score1 = self.alpha * self.batch_pairwise_distance(p1, p2_cat)
        score = score1 + self.beta * self.batch_pairwise_distance(p2, p2_cat) # [B, H, H]

        weight, ind = torch.min(score, dim=2) # [B, H]
        index = ind.unsqueeze(-1).expand([-1, -1, C]) # [B, H, C]
        sel_p2 = torch.gather(p2_cat, dim=1, index=index) # [B, H, C]

        loss = self.criterion(p1, sel_p2)

        return loss


class GBBLoss(BBLoss):

    def __init__(self, alpha=1, beta=1, ksize=3, pad=0, stride=3, dist_norm='l2', criterion='l1'):
        self.batched_gram_matrix = torch.vmap(self.gram_matrix)

        super().__init__(alpha, beta, ksize, pad, stride, dist_norm, criterion)

    def gram_matrix(self, input: Tensor) -> Tensor:
        """ from: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html"""
        d, c = input.size()
        features = input.view(c, d)
        G = torch.mm(features, features.t())

        return G.div(c * d)

    def forward(self, x, gt):
        p1 = F.unfold(x, kernel_size=self.ksize, padding=self.pad, stride=self.stride)
        B, C, H = p1.size()
        p1 = p1.permute(0, 2, 1).contiguous() # [B, H, C]

        p2 = F.unfold(gt, kernel_size=self.ksize, padding=self.pad, stride=self.stride)
        p2 = p2.permute(0, 2, 1).contiguous() # [B, H, C]

        gt_2 = F.interpolate(gt, scale_factor=0.5, mode='bicubic', align_corners = False)
        p2_2 = F.unfold(gt_2, kernel_size=self.ksize, padding=self.pad, stride=self.stride)
        p2_2 = p2_2.permute(0, 2, 1).contiguous() # [B, H, C]

        gt_4 = F.interpolate(gt, scale_factor=0.25, mode='bicubic',align_corners = False)
        p2_4 = F.unfold(gt_4, kernel_size=self.ksize, padding=self.pad, stride=self.stride)
        p2_4 = p2_4.permute(0, 2, 1).contiguous() # [B, H, C]
        p2_cat = torch.cat([p2, p2_2, p2_4], 1)

        score1 = self.alpha * self.batch_pairwise_distance(p1, p2_cat)
        score = score1 + self.beta * self.batch_pairwise_distance(p2, p2_cat) # [B, H, H]

        weight, ind = torch.min(score, dim=2) # [B, H]
        index = ind.unsqueeze(-1).expand([-1, -1, C]) # [B, H, C]
        sel_p2 = torch.gather(p2_cat, dim=1, index=index) # [B, H, C]

        loss = self.criterion(p1, sel_p2)

        return loss