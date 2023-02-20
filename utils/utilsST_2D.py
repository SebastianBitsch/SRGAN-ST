#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions workshop material on structure tensor.

Created on Tue Feb  2 17:15:05 2021

@author: vand, edited by QIM members
"""


import numpy as np
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
from structure_tensor import eig_special_2d, structure_tensor_2d

#%% Plotting functions

def plot_orientations(ax, dim, vec, s = 5):
    """ Helping function for adding orientation-quiver to the plot.
    Arguments: plot axes, image shape, orientation, arrow spacing.
    """
    vx = vec[0].reshape(dim)
    vy = vec[1].reshape(dim)
    xmesh, ymesh = np.meshgrid(np.arange(dim[0]), np.arange(dim[1]), indexing='ij')
    ax.quiver(ymesh[s//2::s,s//2::s],xmesh[s//2::s,s//2::s],vy[s//2::s,s//2::s],vx[s//2::s,s//2::s],color='r',angles='xy')
    ax.quiver(ymesh[s//2::s,s//2::s],xmesh[s//2::s,s//2::s],-vy[s//2::s,s//2::s],-vx[s//2::s,s//2::s],color='r',angles='xy')


def polar_histogram(ax, distribution, cmap = 'hsv'):
    """ Helping function for producing polar histogram.
    Arguments: plot axes, oriantation distribution, colormap.
    """
    N = distribution.size
    bin_centers_full = (np.arange(2*N)+0.5)*np.pi/N # full circle (360 deg)
    distribution_full = np.r_[distribution,distribution]/max(distribution) # full normalized distribution
    x = np.r_[distribution_full*np.cos(bin_centers_full),0]
    y = np.r_[distribution_full*np.sin(bin_centers_full),0]
    triangles = np.array([(i, np.mod(i-1,2*N), 2*N) for i in range(2*N)]) # triangles[0] is symmetric over 0 degree
    triangle_centers_full = (np.arange(2*N))*np.pi/N # a triangle covers area BETWEEN two bin_centers
    triangle_colors = np.mod(triangle_centers_full, np.pi)/np.pi # from 0 to 1-(1/2N)
    ax.tripcolor(y, x, triangles, facecolors=triangle_colors, cmap=cmap, vmin = 0.0, vmax = 1.0)
    ax.set_aspect('equal')
    ax.set_xlim([-1,1])
    ax.set_ylim([1,-1])   


def solve_flow(S):
    """ Solving 1D optic flow, returns LLS optimal x for flow along y axis
        ( x is a solution to S[0]*x=S[2] )
    Arguments:
        S: an array with shape (3,N) containing 2D structure tensor
    Returns:
        x: an array with shape (1,N) containing x components of the flow
    Author: vand@dtu.dk, 2019
    """
    aligned = S[0]==0 # 0 or inf solutions
    x = np.zeros((1,S.shape[1]*S.shape[2]))
    x[0,~aligned.flatten()] = - S[2,~aligned]/S[0,~aligned]
    return x # returning shape (1,N) array for consistancy with 3D case

#%% Wrapping functions for user-friendliness

def st_and_plot(sigma,rho,image):
    
    # compute structure tensor 
    S = structure_tensor_2d(image.astype('float'), sigma, rho)
    val, vec = eig_special_2d(S) 
    
    #Compute preferencial orientation angles
    angle = np.arctan2(vec[1], vec[0])/np.pi 
    orientation_image = angle.reshape(image.shape) 

    # compute anisotropy
    anisotropy = (1-val[0]/val[1]).reshape(image.shape)
    
    #Plot results
    fig, ax = plt.subplots(1, 4, figsize=(20,5), sharex=True, sharey=True)
    
    ax[0].imshow(plt.cm.gray(image))
    ax[0].set_title('Image')
    
    ax[1].imshow(plt.cm.gray(image)*plt.cm.hsv(orientation_image))
    ax[1].set_title('Orientation on image')
    
    ax[2].imshow(anisotropy*image)
    ax[2].set_title('Anisotropy on image')
    
    ax[3].imshow(plt.cm.gray(image)*plt.cm.gray(anisotropy)*plt.cm.hsv(orientation_image))
    ax[3].set_title('Orientation and anisotropy on image')
    
    plt.show()   
    

def st_and_hists(sigma, rho, filename):
    
    N = 180 # number of angle bins for orientation histogram
    
    # computation
    image = skimage.io.imread(filename)
    if len(image.shape) == 3:
        image = np.mean(image[:,:,0:3],axis=2).astype(np.uint8)
    S = structure_tensor_2d(image.astype('float'), sigma, rho)
    val,vec = eig_special_2d(S)
    angles = np.arctan2(vec[1], vec[0]) # angles from 0 to pi
    distribution = np.histogram(angles, bins=N, range=(0.0, np.pi))[0]
    
    # visualisation: images
    figsize = (10,5)
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    ax[0].imshow(image,cmap=plt.cm.gray)
    ax[0].set_title('Input image')
    orientation_st_rgba = plt.cm.hsv((angles/np.pi).reshape(image.shape))
    ax[1].imshow(plt.cm.gray(image)*orientation_st_rgba)
    ax[1].set_title('Orientation as color on image')
    
    # visualisation: histograms
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    bin_centers = (np.arange(N)+0.5)*np.pi/N # halp circle (180 deg)
    colors = plt.cm.hsv(bin_centers/np.pi)
    ax[0].bar(bin_centers, distribution, width = np.pi/N, color = colors)
    ax[0].set_xlabel('angle')
    ax[0].set_xlim([0,np.pi])
    ax[0].set_aspect(np.pi/ax[0].get_ylim()[1])
    ax[0].set_xticks([0,np.pi/2,np.pi])
    ax[0].set_xticklabels(['0','pi/2','pi'])
    ax[0].set_ylabel('count')
    ax[0].set_title('Histogram over angles')
    polar_histogram(ax[1], distribution)
    ax[1].set_title('Polar histogram')
    plt.show()


def cart2sph(x,y,z):
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    #r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation


def histogramSphere(eigVec, nBin):
    
    # Convert eigenvectors from xyz to directions on sphere (azimuth and elevation)
    sphDir = np.empty([2,eigVec.shape[1]], dtype='float')
    for i in range(eigVec.shape[1]):
        sphDir[0,i], sphDir[1,i] = cart2sph(eigVec[0,i],eigVec[1,i],eigVec[2,i])

    # Define uv-histogram (edges)
    cAz  = np.linspace(-np.pi,np.pi,nBin[0]+1)
    cEle = np.linspace(-np.pi/2,np.pi/2,nBin[1]+1)
    
    # Define bin center:
    binC_az = (cAz[:-1] + cAz[1:]) / 2
    binC_ele = (cEle[:-1] + cEle[1:]) / 2
    
    # Area of bins (on the sphere):
    binArea = np.outer((cAz[:-1] - cAz[1:]), \
                  np.sign(np.cos(cEle[:-1])) * np.sin(cEle[:-1]) - \
                  np.sign(np.cos(cEle[1:])) * np.sin(cEle[1:]) )
    
    # Count stats: 
    binCount = np.histogram2d(sphDir[0,:], sphDir[1,:], [cAz, cEle], density=None)[0]
    
    # Normalization:
    binVal = np.empty(binCount.shape,dtype=float)
    totalCount = np.sum(binCount)
    binIdx = np.logical_and(binCount > 1, binArea > 0.0001/np.prod(nBin))
    
    # only 'pdf' for now:
    binVal[binIdx] = binCount[binIdx] / (totalCount * binArea[binIdx]) # area weighting
    
    return binVal, binC_az, binC_ele