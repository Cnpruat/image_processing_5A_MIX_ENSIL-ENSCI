#%% 

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Noyau moyenneur
K = np.ones((21,21), np.float32)/(21**2)
lena = cv2.imread("./Image_files/lena.png")
# Convolution 2D
out = cv2.filter2D(src=lena, ddepth=-1, kernel=K)
plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
plt.show()
# Noyau Gaussien
sigma = 5
k_size = int(2*np.ceil(3*sigma)+1)
gauss_1d = cv2.getGaussianKernel(k_size, sigma)
K_gauss = gauss_1d @ gauss_1d.T # Produit matriciel pour noyau 2D
out_g = cv2.filter2D(lena, -1, K_gauss)
plt.imshow(cv2.cvtColor(out_g, cv2.COLOR_BGR2RGB))
plt.show()
# %%
