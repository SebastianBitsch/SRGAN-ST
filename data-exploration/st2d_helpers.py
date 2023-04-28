import numpy as np
import scipy.ndimage

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