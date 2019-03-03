import cv2
import numpy as np 
import scipy as sc
from matplotlib import  pyplot as plt

def getPdf(given_vec):
	given_h ,_ = np.histogram(given_vec,bins=256)
	given_pdf = given_h.astype(float)/sum(given_h)
	
	return given_pdf

def getCdf(given_pdf):
	given_cdf = given_pdf
	for i in range(1,256):
	    given_cdf[i] = given_cdf[i-1] + given_cdf[i]
	    
	return given_cdf

def histSpec(given_cdf, spec_cdf, given_img):

	new = np.ndarray(shape=np.shape(given_img), dtype=float, order='C')
	index = np.ones(256, order='C')
	
	for i in range(0,256):
		index[i] = spec_cdf[np.argmin(abs(spec_cdf - (given_cdf[i]*np.ones(256,order='C'))))]


	for i in range(0,256):
	    for j in range(0,512):
			new[i,j] = index[given_img[i,j]]

	return new

#Importing images as greyscale
given_img = cv2.imread("givenhist.jpg",0)
spec_img = cv2.imread("sphist.jpg",0)

intensities_given =  np.reshape(given_img,256*512,order='C').astype(float)
intensities_spec =  np.reshape(spec_img,256*512,order='C').astype(float)
L = int(max(intensities_given)-min(intensities_given))
g_pdf = getPdf(intensities_given)
s_pdf = getPdf(intensities_spec)

fig, (given_plt, spec_plt, new_plt) = plt.subplots(1, 3)
given_plt.plot(np.arange(256), g_pdf)
given_plt.set_title('Given image histogram')
spec_plt.plot(np.arange(256), s_pdf)
spec_plt.set_title('Specified image histogram')

g_cdf = getCdf(g_pdf)* L
s_cdf = getCdf(s_pdf)* L

new = histSpec(g_cdf, s_cdf, given_img)
intensities_new =  np.reshape(new,256*512,order='C').astype(float)

new_pdf = getPdf(intensities_new)
new_plt.plot(np.arange(256), new_pdf)
new_plt.set_title('After histogram specification')

fig, (given_plt, new_img) = plt.subplots(1, 2)
given_plt.imshow(given_img, cmap='gray')
given_plt.set_title('Given Image')
new_img.imshow(new, cmap='gray')
new_img.set_title('After histogram specification')

plt.show()
cv2.waitKey(0)

