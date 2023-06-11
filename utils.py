import random
import cv2
import math
from collections import OrderedDict

import torch
from torch import nn
import numpy as np

from torchvision.utils import make_grid


def init_random_seed(seed:int = 0) -> None:
    """ Init random seed to make results reproducible"""
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = True # See: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/2
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_state_dict(model: nn.Module, state_dict: dict) -> nn.Module:
    """
    Load model weights and parameters of a model state-dict.
    This is essentially a wrapper around the inbuilt torch.load_state_dict() to account for a missing
    feature in PyTorch at the moment where state dicts of compiled models cant be loaded in using the
    load_state_dict function. See: https://github.com/pytorch/pytorch/issues/101107#issuecomment-1542688089

    The layernames in a compiled model are prepended with _orig_mod. - we just need to remove this
    """

    model_is_compiled = "_orig_mod" in list(state_dict.keys())[0]

    # Process parameter dictionary
    model_state_dict = model.state_dict()
    new_state_dict = OrderedDict()

    # Check if the model has been compiled
    for layer_name, weights in state_dict.items():

        if model_is_compiled:
            name = layer_name[10:]
        else:
            name = layer_name
        new_state_dict[name] = weights
    state_dict = new_state_dict

    # Traverse the model parameters, load the parameters in the pre-trained model into the current model
    new_state_dict = {k: v for k, v in state_dict.items() if
                      k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}

    # update model parameters
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)

    return model


def tensor2img(tensor:torch.Tensor, out_type = np.uint8, min_max = (0, 1)) -> np.ndarray:
    """
    Convert a tensor of a given shape to a numpy array that can be visualized
    4D: grid (B, C, H, W), 3D: (C, H, W), 2D: (H, W)
    """
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])

    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), padding=0, normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))

    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()

    return img_np.astype(out_type)


def PSNR(img1:np.ndarray, img2:np.ndarray) -> float:
    """
    Calculate the PSNR value between two images
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    return 20 * math.log10(255.0 / math.sqrt(mse))


def SSIM(img1:np.ndarray, img2:np.ndarray) -> float:
    """
    Calculate the SSIM score between two images
    https://en.wikipedia.org/wiki/Structural_similarity
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def bgr2ycbcr(img:np.ndarray, only_y:bool = True) -> np.ndarray:
    """
    bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def batch_pairwise_distance(x:torch.FloatTensor, y:torch.FloatTensor=None, dist_norm:str = 'l1') -> torch.FloatTensor:
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

        dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
        # Ensure diagonal is zero if x=y
        if y is None:
            dist = dist - torch.diag_embed(torch.diagonal(dist, dim1=-2, dim2=-1), dim1=-2, dim2=-1)
        dist = torch.clamp(dist, 0.0, np.inf)
    else:
        raise NotImplementedError('%s norm has not been supported.' % dist_norm)

    return dist


def get_gaussian_kernel(sigma, also_dg=False, radius=None):
    # order only 0 or 1
    
    if radius is None:
        radius = max(int(4*sigma + 0.5), 1)  # similar to scipy _gaussian_kernel1d but never smaller than 1
    x = torch.arange(-radius, radius+1)
    
    sigma2 = (sigma * sigma) + 1e-12
    phi_x = torch.exp(-0.5 / sigma2 * x**2)
    phi_x = phi_x / phi_x.sum()
    
    if also_dg:
        return phi_x.cuda(), (phi_x * -x / sigma2).cuda()
    else:
        return phi_x.cuda()
    


def structure_tensor(im:torch.FloatTensor, sigma:float = 1, rho:float = 10) -> torch.FloatTensor:
    '''
    Image is (1,H,W) torch tensor, st is (3,H,W) torch tensor.
    '''
    g, dg = get_gaussian_kernel(sigma, also_dg=True)
    h = (1, 1, -1, 1)
    w = (1, 1, 1, -1)
    Ix = torch.nn.functional.conv2d(im.unsqueeze(0), dg.reshape(h), padding='same')
    Ix = torch.nn.functional.conv2d(Ix, g.reshape(w), padding='same')
    Iy = torch.nn.functional.conv2d(im.unsqueeze(0), g.reshape(h), padding='same')
    Iy = torch.nn.functional.conv2d(Iy, dg.reshape(w), padding='same')
    
    k = get_gaussian_kernel(rho)
    Jxx = torch.nn.functional.conv2d(Ix ** 2, k.reshape(h), padding='same')
    Jxx = torch.nn.functional.conv2d(Jxx, k.reshape(w), padding='same')
    Jyy = torch.nn.functional.conv2d(Iy ** 2, k.reshape(h), padding='same')
    Jyy = torch.nn.functional.conv2d(Jyy, k.reshape(w), padding='same')
    Jxy = torch.nn.functional.conv2d(Ix * Iy, k.reshape(h), padding='same')
    Jxy = torch.nn.functional.conv2d(Jxy, k.reshape(w), padding='same')
    
    S = torch.cat((Jxx.squeeze(0), Jyy.squeeze(0), Jxy.squeeze(0)), dim=0)
    return S


def normalize(S: torch.FloatTensor, eps:float = 1e-12) -> torch.FloatTensor:
    """ Normalize a 2x2 matrice using the determinant """
    d = S[0]*S[1] - S[2]**2
    return S / torch.sqrt(d + eps)


def compute_invS1xS2(S1, S2, _normalize = True):
    ''' Pixelwise inv(S1)*S2 for two symmetric 2x2 matrices.'''
    
    if _normalize:
        S1 = normalize(S1)
        S2 = normalize(S2)   
    A = (S1[1]*S2[0] - S1[2]*S2[2])  # Element M_11
    B = (S1[0]*S2[1] - S1[2]*S2[2])  # Element M_22
    C = (S1[1]*S2[2] - S1[2]*S2[1])  # Element M_12
    D = (S1[0]*S2[2] - S1[2]*S2[0])  # Element M_21
    out = torch.stack((A, B, C, D), dim=1)
    out = out.permute(1, 0, 2)
    return out


def compute_eigenvalues(M: torch.FloatTensor, eps:float = 1e-12) -> torch.FloatTensor:
    ''' Pixelwise eigenvalues of 2x2 matrix.'''
    
    ApB = M[0] + M[1]
    discriminant = ApB ** 2 - 4 * (M[0] * M[1] - M[2] * M[3]) # THIS SHOULD ALWAYS BE > 0
    discriminant = torch.clamp(discriminant, min = 0 + eps)  # TODO, better way of handling this?
    r = torch.sqrt(discriminant)
    l1 = 0.5*(ApB - r)
    l2 = 0.5*(ApB + r)
    return torch.stack((l1, l2), dim=1)
    

def compute_distance(L: torch.FloatTensor, eps:float = 1e-12) -> torch.FloatTensor:
    ''' Pixelwise Riemannian distance from eigenvalues'''
    
    # i = (L<1).sum()
    #if i>0:
    # print(f'We have {i} eigenvalues smaller than 1!')
    L = torch.clamp(L, min=1)  # TODO, better way of handling this?
    L = torch.log(L)
    L = L**2
    L = L.permute(1,0,2) # OBS
    d = L.sum(dim=0)
    return torch.sqrt(d + eps)