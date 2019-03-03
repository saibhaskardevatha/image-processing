# Harris Corner Detection
# Step 1 - Compute Derivatives <br>
# Step 2 - Compute Matrix M (Structure Tensor Matrix) <br>
# Step 3 - Compute Corner Response function (R) <br>
# Step 4 - Threshold R <br>
# Step 5 - Non Maximal Supression <br>


#Import necessary libraries
import numpy as np
from matplotlib import pyplot as plt
import statistics
import cv2

#Import image
img_color = cv2.imread('IITG.jpg', -1)
img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

#Initialising Variables required.
r = np.zeros(shape = (526,796))
height = img.shape[0]
width = img.shape[1]
mat = np.zeros(shape = (2,2))

#Compute Derivatives (dy - Vertical Gradient, dx - Horizontal Gradient)
dx = np.zeros(img.shape, dtype=float)
dy = np.zeros(img.shape, dtype=float)

for i in range(0,528):
    dx[i,:] = img[i,:] - img[i+1,:]
for j in range(0,798):
    dy[:,j] = img[:,j] - img[:,j+1]

Ixx = dx**2
Iyy = dy**2
Ixy = dx*dy

#Compute M and R
alpha = 0.06

for i in range(2,height-2):
    for j in range(2, width-2):
        mat[0,0] = Ixx[i-2:i+2,j-2:j+2].sum() 
        mat[1,1] = Iyy[i-2:i+2,j-2:j+2].sum()
        mat[0,1] = Ixy[i-2:i+2,j-2:j+2].sum()
        mat[1,0] = mat[0,1]
        
        r[i-2,j-2] = mat[0,0]*mat[1,1] - alpha*(mat[0,1]*mat[1,0])

#Threshold R and Non Maximal Supression
r_pad = np.pad(r,2,'constant')

#K = 3.5
corner = np.zeros(shape = r.shape)
threshold = 3.5*np.mean(r)

for i in range(2,r.shape[0]):
    for j in range(2,r.shape[1]):
        if r_pad[i,j]>threshold:
            if r_pad[i,j] >= np.max(r_pad[i-2:i+2,j-2:j+2]):
                corner[i-2,j-2] = 255
            else:
                corner[i-2,j-2] = 0

#Plot points of Non Maximal Supression
corners = np.pad(corner,2,'constant')

#Plot corners on IITG image
x = []
y = []
for i in range(0,799):
    for j in range(0,529):
        if corners[j][i] == 255:
            x.append(i)
            y.append(j)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.imshow(img_color)
ax1.scatter(x,y,c='red',marker='x')
ax1.set_title('Alpha = 0.06, K = 3.5')

#K = 4
corner = np.zeros(shape = r.shape)
threshold = 4*np.mean(r)

for i in range(2,r.shape[0]):
    for j in range(2,r.shape[1]):
        if r_pad[i,j]>threshold:
            if r_pad[i,j] >= np.max(r_pad[i-2:i+2,j-2:j+2]):
                corner[i-2,j-2] = 255
            else:
                corner[i-2,j-2] = 0

#Plot points of Non Maximal Supression
corners = np.pad(corner,2,'constant')

#Plot corners on IITG image
x = []
y = []
for i in range(0,799):
    for j in range(0,529):
        if corners[j][i] == 255:
            x.append(i)
            y.append(j)

ax2.imshow(img_color)
ax2.scatter(x,y,c='red',marker='x')
ax2.set_title('Alpha = 0.06, K = 4')
plt.show()