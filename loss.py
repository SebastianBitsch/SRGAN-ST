import torch
import numpy as np

from torch import nn, Tensor

import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from structure_tensor import structure_tensor_2d





class ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.

    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.
     """

    def __init__(self, extraction_layers: dict[str, float], device) -> None:
        """
        Content loss (in SRGAN) / Perceptual loss (in GramGAN).
        Follows the method outlined in GramGAN paper for computing a loss from the activation layer 
        in the pre-trained VGG19 network.
        
        Parameters
            extraction_layers (dict): A dict of layer
        """
        super(ContentLoss, self).__init__()

        # Get the name of the specified feature extraction node
        self.extraction_layers = extraction_layers
        self.device = device

        # Load the VGG19 model trained on the ImageNet dataset.
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)

        # Extract the thirty-sixth layer output in the VGG19 model as the content loss.
        self.feature_extractor = create_feature_extractor(model, list(extraction_layers))

        # set to validation mode
        self.feature_extractor.eval()

        # This is the VGG model preprocessing method of the ImageNet dataset.
        # The mean and std of ImageNet. See: https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
        self.normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

        # Freeze model parameters.
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False

    def __repr__(self):
        return "ContentLoss()"

    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor) -> Tensor:
        # Standardized operations
        sr_tensor = self.normalize(sr_tensor)
        gt_tensor = self.normalize(gt_tensor)

        # Find the feature map difference between the two images
        loss = torch.tensor(0.0, device = self.device)
        for name, weight in self.extraction_layers.items():
            sr_feature = self.feature_extractor(sr_tensor)[name]
            gt_feature = self.feature_extractor(gt_tensor)[name]

            # loss += weight * torch.linalg.vector_norm(sr_feature - gt_feature, ord = 1)
            loss += weight * F.mse_loss(sr_feature, gt_feature)
        
        return loss



class EuclidLoss(nn.Module):
    """  """

    def __init__(self) -> None:
        super(EuclidLoss, self).__init__()
        self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor) -> Tensor:        
        sr_tensor = torch.flatten(sr_tensor)
        gt_tensor = torch.flatten(gt_tensor)
        
        loss = self.pdist(sr_tensor, gt_tensor)

        return loss

def batch_pairwise_distance(x, y=None, dist_norm = 'l1'):
    '''
    Input: x is a BxNxd matrix
            y is an optional BxMxd matirx
    Output: dist is a BxNxM matrix where dist[b,i,j] is the square norm between x[b,i,:] and y[b,j,:]
            if y is not given then use 'y=x'.
    i.e. dist[b,i,j] = ||x[b,i,:]-y[b,j,:]||^2
    '''
    B, N, d = x.size()
    if dist_norm == 'l1':
        x_norm = x.view(B, N, 1, d)
        if y is not None:
            y_norm = y.view(B, 1, -1, d)
        else:
            y_norm = x.view(B, 1, -1, d)
        dist = torch.abs(x_norm - y_norm).sum(dim=3)
    elif dist_norm == 'l2':
        x_norm = (x ** 2).sum(dim=2).view(B, N, 1)
        if y is not None:
            M = y.size(1)
            y_t = torch.transpose(y, 1, 2)
            y_norm = (y ** 2).sum(dim=2).view(B, 1, M)
        else:
            y_t = torch.transpose(x, 1, 2)
            y_norm = x_norm.view(B, 1, N)

        print(x_norm.shape, y_norm.shape, x.shape, y_t.shape)
        dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
        # Ensure diagonal is zero if x=y
        if y is None:
            dist = dist - torch.diag_embed(torch.diagonal(dist, dim1=-2, dim2=-1), dim1=-2, dim2=-1)
        dist = torch.clamp(dist, 0.0, np.inf)
        # dist = torch.sqrt(torch.clamp(dist, 0.0, np.inf) / d)
    else:
        raise NotImplementedError('%s norm has not been supported.' % dist_norm)

    return dist


class BestBuddyLoss(nn.Module):
    """ https://github.com/dvlab-research/Simple-SR/blob/08c71e9e46ba781df50893f0476ecd0fc004aa45/utils/loss.py#L54 """
    def __init__(self, alpha=1.0, beta=1.0, ksize=3, pad=0, stride=3, dist_norm='l2', criterion='l1'):
        super(BestBuddyLoss, self).__init__()
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

        print("bb",p1.shape, p2_cat.shape)
        score1 = self.alpha * batch_pairwise_distance(p1, p2_cat, self.dist_norm)
        score = score1 + self.beta * batch_pairwise_distance(p2, p2_cat, self.dist_norm) # [B, H, H]

        weight, ind = torch.min(score, dim=2) # [B, H]
        index = ind.unsqueeze(-1).expand([-1, -1, C]) # [B, H, C]
        sel_p2 = torch.gather(p2_cat, dim=1, index=index) # [B, H, C]

        loss = self.criterion(p1, sel_p2)

        return loss

class GramLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, ksize=3, dist_norm='l2', criterion='l1'):
        super(GramLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ksize = ksize
        self.dist_norm = dist_norm

        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss(reduction='mean')
        elif criterion == 'l2':
            self.criterion = torch.nn.L2loss(reduction='mean')
        else:
            raise NotImplementedError('%s criterion has not been supported.' % criterion)

    def gram_matrix(self, input):
        b, c, d = input.size() # 3,3,3
        features = input.view(b, c * d)
        G = torch.mm(features, features.t())
        return G.div(b * c * d)

    def do_work(self, x):
        """
        A lot of careful gymnastics to unfold the batch of images into nice patches and take the 
        gram matrix of every one.
        Doesnt support padding or stride. too hard to do tbh
        """
        B,_,_,_ = x.shape
        x = x.unfold(1, self.ksize, self.ksize).unfold(2, self.ksize, self.ksize).unfold(3, self.ksize, self.ksize)     #-> torch.Size([16, 1, 64, 64, 3, 3, 3])
        x = x.squeeze()                                             #-> torch.Size([16, 64, 64, 3, 3, 3])
        x = x.reshape(B, -1, self.ksize, self.ksize, self.ksize)    #-> torch.Size([16, 4096, 3, 3, 3])

        batched_gram = torch.func.vmap(torch.func.vmap(self.gram_matrix))

        x = batched_gram(x)                                 #-> torch.Size([16, 4096, 3, 3]) uh ja
        x = x.reshape(B, -1, self.ksize * self.ksize)       #-> torch.Size([16, 4096, 9])
        return x

    def forward(self, x, gt):
        # p1 = F.unfold(x, kernel_size=self.ksize, padding=self.pad, stride=self.stride) # ([16, 27, 4096])
        # B, C, H = p1.size()
        # p1 = p1.permute(0, 2, 1).contiguous() # ([16, 4096, 27])
        p1 = self.do_work(x)
        _, C, _ = p1.size()

        # p2 = F.unfold(gt, kernel_size=self.ksize, padding=self.pad, stride=self.stride)
        # p2 = p2.permute(0, 2, 1).contiguous() # [B, H, C]
        p2 = self.do_work(gt)

        gt_2 = F.interpolate(gt, scale_factor=0.5, mode='bicubic', align_corners = False)
        # p2_2 = F.unfold(gt_2, kernel_size=self.ksize, padding=self.pad, stride=self.stride)
        # p2_2 = p2_2.permute(0, 2, 1).contiguous() # [B, H, C]
        p2_2 = self.do_work(gt_2)

        gt_4 = F.interpolate(gt, scale_factor=0.25, mode='bicubic', align_corners = False)
        # p2_4 = F.unfold(gt_4, kernel_size=self.ksize, padding=self.pad, stride=self.stride)
        # p2_4 = p2_4.permute(0, 2, 1).contiguous() # [B, H, C]
        p2_4 = self.do_work(gt_4)
        p2_cat = torch.cat([p2, p2_2, p2_4], 1)

        print(p1.shape, p2_cat.shape)
        score1 = self.alpha * batch_pairwise_distance(p1, p2_cat, self.dist_norm)
        score = score1 + self.beta * batch_pairwise_distance(p2, p2_cat, self.dist_norm) # [B, H, H]

        weight, ind = torch.min(score, dim=2) # [B, H]
        index = ind.unsqueeze(-1).expand([-1, -1, C]) # [B, H, C]
        sel_p2 = torch.gather(p2_cat, dim=1, index=index) # [B, H, C]

        loss = self.criterion(p1, sel_p2)


class STLoss(BestBuddyLoss):

    def __init__(self, alpha=1, beta=1, ksize=3, pad=0, stride=3, dist_norm='l2', criterion='l1'):
        super().__init__(alpha, beta, ksize, pad, stride, dist_norm, criterion)

    def st_mat(self, x):
        """
        Computes the structure tensor

        in: torch.Size([16, 3, 96, 96])
        out: torch.Size([16, 3, 3])
        """
        sigma = 0.4 # config.ST.sigma eller lign
        rho = 2.5   # config.ST.rho

        # Extracts the batchsize
        batch_size = x.shape[0]
        # Grayscale image since structure tensors can't handle colors
        x = transforms.Grayscale()(x)
        result = []
        for i in range(batch_size):
            # x[:,0,...] since structure tensor func needs (height, width)
            image = x[:,0,...][i].cpu().numpy()
            S = structure_tensor_2d(image, sigma, rho)
            result.append(torch.from_numpy(S))
        return torch.stack(result)

    def forward(self, x, gt):
        """ https://github.com/dvlab-research/Simple-SR/blob/master/utils/loss.py#L94 """
        # x and gt: torch.Size([16, 3, 96, 96])

        # Get the gram matrix of the estimated patch and calculate the candidate patches
        g_x = self.st_mat(x)
        g_gt = self.st_mat(gt)
        g_gt2 = self.st_mat(F.interpolate(gt, scale_factor=1/2, mode="bicubic"))
        g_gt4 = self.st_mat(F.interpolate(gt, scale_factor=1/4, mode="bicubic"))
        # TODO: Calculate more candidate patches by affine transformations

        # Combine all candidates
        gt_cat = torch.cat([g_gt, g_gt2, g_gt4], 1) # torch.Size([16, 9, 3])

        # Use Eq. 2
        score_a = self.alpha * self.batch_pairwise_distance(g_gt, gt_cat)
        score_b = self.beta * self.batch_pairwise_distance(g_x, gt_cat)
        score = score_a + score_b

        # Complicated way of taking argmin to get the best patch
        weight, ind = torch.min(score, dim=2) # [B, H]
        index = ind.unsqueeze(-1).expand([-1, -1, 3]) 
        best_patch = torch.gather(gt_cat, dim=1, index=index) # torch.Size([16, 3, 3])

        # Use Eq. 4
        loss = self.criterion(g_x, best_patch)

        return loss
