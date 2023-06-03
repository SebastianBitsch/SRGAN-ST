import torch
from collections import defaultdict
from torch import nn, Tensor
from functools import reduce

import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from model import Discriminator

from utils import batch_pairwise_distance, structure_tensor, normalize, compute_invS1xS2, compute_eigenvalues, compute_distance


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

        # This is the VGG model preprocessing method of the ImageNet dataset.
        # The mean and std of ImageNet. See: https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
        self.normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

        # Freeze model parameters.
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False

        # Set to validation mode
        self.feature_extractor.eval()


    def __repr__(self):
        return "ContentLoss()"

    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor) -> Tensor:
        # Standardized operations
        sr_tensor = self.normalize(sr_tensor)
        gt_tensor = self.normalize(gt_tensor)

        sr_feature = self.feature_extractor(sr_tensor)
        gt_feature = self.feature_extractor(gt_tensor)

        # Find the feature map difference between the two images
        loss = torch.tensor(0.0, device = self.device)
        for name, weight in self.extraction_layers.items():
            loss += weight * F.mse_loss(sr_feature[name], gt_feature[name])
        
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

        score1 = self.alpha * batch_pairwise_distance(p1, p2_cat, self.dist_norm)
        score = score1 + self.beta * batch_pairwise_distance(p2, p2_cat, self.dist_norm) # [B, H, H]

        weight, ind = torch.min(score, dim=2) # [B, H]
        index = ind.unsqueeze(-1).expand([-1, -1, C]) # [B, H, C]
        sel_p2 = torch.gather(p2_cat, dim=1, index=index) # [B, H, C]

        loss = self.criterion(p1, sel_p2)

        return loss



class GramLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, ksize=3, dist_norm='l2', criterion='l1'):
        """ Note: image size must be devisable by ksize """
        super(GramLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ksize = ksize
        self.dist_norm = dist_norm

        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss(reduction='mean')
        elif criterion == 'l2' or criterion == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='mean')
        else:
            raise NotImplementedError('%s criterion has not been supported.' % criterion)

    def gram_matrix(self, input):
        b, c, d = input.size()
        features = input.view(b, c * d)
        G = torch.mm(features, features.t())
        return G.div(b * c * d)

    def compute_patches(self, x):
        """
        A lot of careful gymnastics to unfold the batch of images into nice patches and take the 
        gram matrix of every one.
        Doesnt support padding or stride. too hard to do tbh
        """
        B,_,_,_ = x.shape
        x = x.unfold(1, 3, self.ksize).unfold(2, self.ksize, self.ksize).unfold(3, self.ksize, self.ksize)     #-> torch.Size([16, 1, 64, 64, 3, 3, 3])
        x = x.squeeze()                                             #-> torch.Size([16, 64, 64, 3, 3, 3])
        x = x.reshape(B, -1, 3, self.ksize, self.ksize)    #-> torch.Size([16, 4096, 3, 3, 3])

        batched_gram = torch.func.vmap(torch.func.vmap(self.gram_matrix))

        x = batched_gram(x)                                 #-> torch.Size([16, 4096, 3, 3]) uh ja
        x = x.reshape(B, -1, self.ksize * self.ksize)       #-> torch.Size([16, 4096, 9])
        return x

    def forward(self, x, gt):
        p1 = self.compute_patches(x)
        _, _, W = p1.size()

        p2 = self.compute_patches(gt)

        gt_2 = F.interpolate(gt, scale_factor=0.5, mode='bicubic', align_corners = False)
        p2_2 = self.compute_patches(gt_2)

        gt_4 = F.interpolate(gt, scale_factor=0.25, mode='bicubic', align_corners = False)
        p2_4 = self.compute_patches(gt_4)
        p2_cat = torch.cat([p2, p2_2, p2_4], 1)

        score1 = self.alpha * batch_pairwise_distance(p1, p2_cat, self.dist_norm)
        score = score1 + self.beta * batch_pairwise_distance(p2, p2_cat, self.dist_norm) # [B, H, H]

        _, ind = torch.min(score, dim=2) # [B, H]
        index = ind.unsqueeze(-1).expand([-1, -1, W]) # [B, H, C]
        sel_p2 = torch.gather(p2_cat, dim=1, index=index) # [B, H, C]

        loss = self.criterion(p1, sel_p2)

        return loss



class DiscriminatorFeaturesLoss(nn.Module):
    
    def __init__(self, discriminator: Discriminator, extraction_layers: dict[int, float], config) -> None:
        super(DiscriminatorFeaturesLoss, self).__init__()
        
        # Add hooks to all the layers which will update the self.activations value whenever forwards is called on the model
        for layer_name, _ in extraction_layers.items():
            layer = self.get_layer_by_name(discriminator, layer_name)
            layer.register_forward_hook(self.get_activation(layer_name))

        self.device = config.DEVICE
        self.activations = defaultdict(list)
        self.extraction_layers = extraction_layers


    def get_layer_by_name(self, layer, access_string):
        """ Get a reference to a given layer of a torch network"""
        names = access_string.split(sep='.')
        return reduce(getattr, names, layer)


    def get_activation(self, name):
        def hook(_model, _input, output):
            # input: torch.Size([1, 64, 46, 46])
            self.activations[name].append(output.detach())
        return hook


    def forward(self, _x, _gt):
        """
        Inputs _x and _gt arent used as the values we really need (the activations) will 
        already have been written to self.activations via the webhook. This happens during 
        the forward pass of the discriminator
        """
        loss = torch.tensor(0.0, device=self.device)
        for layer_name, (x, gt) in self.activations.items():
            weight = self.extraction_layers[layer_name]
            # loss += weight * torch.nn.L1Loss(reduction='mean')(x, gt)
            loss += weight * F.mse_loss(x, gt)

        # Clear the dict
        self.activations = defaultdict(list)
        return loss



class PatchwiseStructureTensorLoss(nn.Module):
    def __init__(self, sigma=1, rho=10, alpha=1.0, beta=1.0, ksize=3, dist_norm='l2', criterion='l1'):
        super(PatchwiseStructureTensorLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ksize = ksize
        self.dist_norm = dist_norm
        self.sigma = sigma
        self.rho = rho

        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss(reduction='mean')
        elif criterion == 'l2' or criterion == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='mean')
        else:
            raise NotImplementedError('%s criterion has not been supported.' % criterion)

    def s_norm(self, x):
        """ Compute the structure tensor of a matrix"""
        x = transforms.Grayscale()(x)
        x = structure_tensor(x, sigma=self.sigma, rho=self.rho)
        return normalize(x)

    def compute_patches(self, x):
        """
        A lot of careful gymnastics to unfold the batch of images into nice patches and take the 
        gram matrix of every one.
        Doesnt support padding or stride. too hard to do tbh
        """
        B,_,_,_ = x.shape
        x = x.unfold(1, 3, self.ksize).unfold(2, self.ksize, self.ksize).unfold(3, self.ksize, self.ksize)     #-> torch.Size([16, 1, 64, 64, 3, 3, 3])
        x = x.squeeze()                                             #-> torch.Size([16, 64, 64, 3, 3, 3])
        x = x.reshape(B, -1, 3, self.ksize, self.ksize)    #-> torch.Size([16, 4096, 3, 3, 3])

        batched_gram = torch.func.vmap(torch.func.vmap(self.s_norm))
        x = batched_gram(x)                                 #-> torch.Size([16, 4096, 3, 3])
        x = x.reshape(B, -1, 3 * self.ksize * self.ksize)       #-> torch.Size([16, 4096, 9])
        return x

    def forward(self, x, gt):
        p1 = self.compute_patches(x)
        _, _, W = p1.size()

        p2 = self.compute_patches(gt)

        gt_2 = F.interpolate(gt, scale_factor=0.5, mode='bicubic', align_corners = False)
        p2_2 = self.compute_patches(gt_2)

        gt_4 = F.interpolate(gt, scale_factor=0.25, mode='bicubic', align_corners = False)
        p2_4 = self.compute_patches(gt_4)
        
        p2_cat = torch.cat([p2, p2_2, p2_4], 1)

        score1 = self.alpha * batch_pairwise_distance(p1, p2_cat, self.dist_norm)
        score = score1 + self.beta * batch_pairwise_distance(p2, p2_cat, self.dist_norm) # [B, H, H]

        _, ind = torch.min(score, dim=2) # [B, H]
        index = ind.unsqueeze(-1).expand([-1, -1, W]) # [B, H, C]
        sel_p2 = torch.gather(p2_cat, dim=1, index=index) # [B, H, C]

        loss = self.criterion(p1, sel_p2)

        return loss


class StructureTensorLoss(nn.Module):
    def __init__(self, sigma=1, rho=10):
        super(StructureTensorLoss, self).__init__()
        self.sigma = sigma
        self.rho = rho
    
    
    def st_loss(self, x, gt, normalize = True):
        x = transforms.Grayscale()(x)
        gt = transforms.Grayscale()(gt)

        s_x = structure_tensor(x, sigma=self.sigma, rho=self.rho)
        s_gt = structure_tensor(gt, sigma=self.sigma, rho=self.rho)

        M = compute_invS1xS2(s_x, s_gt, normalize)
        L = compute_eigenvalues(M)
        d = compute_distance(L)
        return d.mean()

    def forward(self, x, gt):
        batched_st_loss = torch.vmap(self.st_loss)
        return batched_st_loss(x, gt).mean()
