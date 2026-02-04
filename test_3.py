#%%

import cv2
import numpy as np
import matplotlib.pyplot as plt

subject = cv2.imread("./Image_files/greenscreen.jpg").astype(float)
plt.figure(figsize=(6,8))
plt.imshow(subject.astype(np.uint8))
# Calcul de l’intensité (Somme R+G+B)
S = np.sum(subject, axis=2)
S[S==0] = 1.0 # Eviter division par zero
# Coordonnée g (Attention: BGR -> G est l’indice 1)
g = subject[:,:,1] / S
mask = g < 0.45
plt.figure(figsize=(6,8))
plt.imshow(mask, cmap="gray")
# Application du masque (Multiplication matricielle)
mask_3d = np.stack([mask]*3, axis=2) # On étend le masque en 3D
res = subject * mask_3d
plt.figure(figsize=(6,8))
plt.imshow(res.astype(np.uint8))
plt.show()
# %%
