#%% 

import cv2
import numpy as np
import matplotlib.pyplot as plt

im = cv2.imread("./Image_files/lena.png")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# Normalisation en flottant (0.0 - 1.0)
im_double = im.astype(float) / 255.0
# Histogramme (aplatir l’image avec ravel)
hist_vals, bins = np.histogram(im_double.ravel(), bins=256, range=(0,1))

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(im_double)
plt.title("Image Originale")
plt.axis("off")

plt.subplot(2,2,3)
plt.plot(hist_vals)
plt.title("Histogramme de l'image originale")


# Etirement de contraste (Percentile)
p2, p98 = np.percentile(im_double, (2, 98))
im_s = np.clip((im_double - p2)/(p98 - p2), 0, 1)

plt.subplot(2,2,2)
plt.imshow(im_s)
plt.title("Image contrastée")
plt.axis("off")

hist_vals_s, bins = np.histogram(im_s.ravel(), bins=256, range=(0,1))

plt.subplot(2,2,4)
plt.plot(hist_vals_s)
plt.title("Histogramme de l'image constrastée")

plt.imsave(f"Res_test/2_01_originale.png", im_double, cmap="gray")
plt.imsave(f"Res_test/2_02_constraste.png", im_s, cmap="gray")
plt.show()
# %%
