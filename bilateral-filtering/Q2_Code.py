import cv2
import numpy as np 
from matplotlib import  pyplot as plt

def gaussian(i, j, k, l, I1, I2, sigma_dom, sigma_range):
	domain_t = np.float64((i-k)**2 + (j-l)**2)/(2.0*(sigma_dom**2))
	range_t = ((np.float64(I1) - np.float64(I2))**2)/(2.0*(sigma_range**2))
	gauss = np.exp(-(domain_t + range_t))
	return gauss

def filter(img, fsize, sigma_dom, sigma_range):
	img = cv2.copyMakeBorder(img,int(fsize/2),int(fsize/2),int(fsize/2),int(fsize/2),cv2.BORDER_REFLECT)
	new_img = np.zeros(shape=np.shape(img), dtype=np.float64, order='C')
	weights = 0.0

	for x in range(int(fsize/2), np.shape(img)[0] - int(fsize/2)):
		for y in range(int(fsize/2) , np.shape(img)[1] - int(fsize/2)):
			for i in range(0, fsize):
				for j in range(0,fsize):
					k = x+i-2
					l = y+j-2
					new_img[x][y] = new_img[x][y] + gaussian(x,y,k,l,img[x][y], img[k][l], sigma_dom, sigma_range) * img[k][l]
					weights = weights + gaussian(x,y,k,l,img[x][y], img[k][l], sigma_dom, sigma_range)
			new_img[x][y] = new_img[x][y]/weights
			weights = 0.0
	return new_img 


img = cv2.imread("spnoisy.jpg",0)

fig, plots = plt.subplots(3, 4)
new = filter(img, 5, 10,10)
plots[0, 0].imshow(new, cmap = 'gray')
new = filter(img, 5, 10,30)
plots[0, 1].imshow(new, cmap = 'gray')
new = filter(img, 5, 10,90)
plots[0, 2].imshow(new, cmap = 'gray')
new = filter(img, 5, 10,200)
plots[0, 3].imshow(new, cmap = 'gray')
new = filter(img, 5, 20,10)
plots[1, 0].imshow(new, cmap = 'gray')
new = filter(img, 5, 20,30)
plots[1, 1].imshow(new, cmap = 'gray')
new = filter(img, 5, 20,90)
plots[1, 2].imshow(new, cmap = 'gray')
new = filter(img, 5, 20,200)
plots[1, 3].imshow(new, cmap = 'gray')
new = filter(img, 5, 30,10)
plots[2, 0].imshow(new, cmap = 'gray')
new = filter(img, 5, 30,30)
plots[2, 1].imshow(new, cmap = 'gray')
new = filter(img, 5, 30,90)
plots[2, 2].imshow(new, cmap = 'gray')
new = filter(img, 5, 30,200)
plots[2, 3].imshow(new, cmap = 'gray')

plt.show()
cv2.waitKey(0)

