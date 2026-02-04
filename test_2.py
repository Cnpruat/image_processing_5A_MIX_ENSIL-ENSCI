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
plt.figure(figsize=(6,8))
# L’image
plt.subplot(2,1,1)
plt.imshow(im_double)
plt.title("Image contrastée")
plt.axis("off")
# L’histogramme
plt.subplot(2,1,2)
plt.plot(hist_vals)
plt.title("Histogramme")


# Etirement de contraste (Percentile)
plt.figure(figsize=(6,8))

plt.subplot(2,1,1)
p2, p98 = np.percentile(im_double, (2, 98))
im_s = np.clip((im_double - p2)/(p98 - p2), 0, 1)
plt.imshow(im_s)

plt.subplot(2,1,2)
hist_vals_s, bins = np.histogram(im_s.ravel(), bins=256, range=(0,1))
plt.plot(hist_vals_s)
plt.title("Histogramme")

plt.show()
# %%
