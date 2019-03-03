import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage.filters import convolve as conv

#n - iterations, l - lambda, k - K
def anisotropic(n, l, k, ax):
    out_img = cv2.imread('lenna-noise.jpg',0)
    out_img = out_img/1.0
    
    #2D Masks
    hN = np.array([[0,1,0],[0,-1,0],[0,0,0]], dtype = np.float64)
    hS = np.array([[0,0,0],[0,-1,0],[0,1,0]], dtype = np.float64)
    hW = np.array([[0,0,0],[1,-1,0],[0,0,0]], dtype = np.float64)
    hE = np.array([[0,0,0],[0,-1,1],[0,0,0]], dtype = np.float64)
    hNW = np.array([[1,0,0],[0,-1,0],[0,0,0]], dtype = np.float64)
    hNE = np.array([[0,0,1],[0,-1,0],[0,0,0]], dtype = np.float64)
    hSW = np.array([[0,0,0],[0,-1,0],[1,0,0]], dtype = np.float64)
    hSE = np.array([[0,0,0],[0,-1,0],[0,0,1]], dtype = np.float64)
    
    for i in range(1,n):
        phiN = conv(out_img,hN)
        phiS = conv(out_img,hS)
        phiE = conv(out_img,hE)
        phiW = conv(out_img,hW)
        phiNW = conv(out_img,hNW)
        phiNE = conv(out_img,hNE)
        phiSW = conv(out_img,hSW)
        phiSE = conv(out_img,hSE)
        
        cN = 1/(1+((phiN/k)**2))
        cS = 1/(1+((phiS/k)**2))
        cE = 1/(1+((phiE/k)**2))
        cW = 1/(1+((phiW/k)**2))
        cNE = 1/(1+((phiNE/k)**2))
        cNW = 1/(1+((phiNW/k)**2))
        cSW = 1/(1+((phiSW/k)**2))
        cSE = 1/(1+((phiSE/k)**2))
        
        out_img += l*(cN*phiN + cS*phiS + cE*phiE + cW*phiW + 0.5*(cNE*phiNE + cSW*phiSW + cNW*phiNW + cSE*phiSE))
    
    ax.imshow(out_img,cmap='gray')
    return out_img


f, (ax1,ax2,ax3) = plt.subplots(1, 3, sharey=True)
img1 = anisotropic(20,0.1,15,ax1)
ax1.set_title("Iterations = 20")
img2 = anisotropic(30,0.1,15,ax2)
ax2.set_title("Iterations = 30")
img3 = anisotropic(40,0.1,15,ax3)
ax3.set_title("Iterations = 40")
plt.show()


def isotropic(n, l, ax):
    out_img = cv2.imread('lenna-noise.jpg',0)

    h=np.array([[0,l,0],[l,1-(4*l),l],[0,l,0]])
    for i in range(1,n):
        out_img = conv(out_img,h)
    
    ax.imshow(out_img, cmap='gray')


f, (ax1,ax2,ax3) = plt.subplots(1, 3, sharey=True)
img1 = isotropic(20,0.15,ax1)
ax1.set_title("Iterations = 20")
img2 = isotropic(30,0.15,ax2)
ax2.set_title("Iterations = 30")
img3 = isotropic(40,0.15,ax3)
ax3.set_title("Iterations = 40")
plt.show()