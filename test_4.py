#%% 

import cv2
import numpy as np
import matplotlib.pyplot as plt

lena = cv2.imread("./Image_files/lena.png")

plt.figure(figsize=(6,9))
plt.subplot(3,1,1)
plt.imshow(cv2.cvtColor(lena, cv2.COLOR_BGR2RGB))
plt.title("Image originale")
plt.axis("off")

# Noyau moyenneur
K = np.ones((21,21), np.float32)/(21**2)
# Convolution 2D
out = cv2.filter2D(src=lena, ddepth=-1, kernel=K)

plt.subplot(3,1,2)
plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
plt.title("Filtre moyenneur")
plt.axis("off")

# Noyau Gaussien
sigma = 5
k_size = int(2*np.ceil(3*sigma)+1)
gauss_1d = cv2.getGaussianKernel(k_size, sigma)
K_gauss = gauss_1d @ gauss_1d.T # Produit matriciel pour noyau 2D
out_g = cv2.filter2D(lena, -1, K_gauss)

plt.subplot(3,1,3)
plt.imshow(cv2.cvtColor(out_g, cv2.COLOR_BGR2RGB))
plt.title("Filtre Gaussien")
plt.axis("off")

plt.imsave(f"Res_test/4_01_originale.png", cv2.cvtColor(lena, cv2.COLOR_BGR2RGB), cmap="gray")
plt.imsave(f"Res_test/4_02_originale.png", cv2.cvtColor(out, cv2.COLOR_BGR2RGB), cmap="gray")
plt.imsave(f"Res_test/4_03_masque.png", cv2.cvtColor(out_g, cv2.COLOR_BGR2RGB), cmap="gray")
plt.show()
# %%
