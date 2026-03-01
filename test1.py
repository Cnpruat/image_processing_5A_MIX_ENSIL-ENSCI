import cv2
import numpy as np
import matplotlib.pyplot as plt

flowers = cv2.imread("./Image_files/flowers8.png")
# Conversion BGR -> RGB pour l’affichage correct
flowers = cv2.cvtColor(flowers, cv2.COLOR_BGR2RGB)
print(flowers.shape, flowers.dtype)

# Conversion RGB -> GRAY pour l’affichage en nuances de gris
grey = cv2.cvtColor(flowers, cv2.COLOR_RGB2GRAY)

plt.subplot(1,2,1)
plt.imshow(flowers, cmap="gray")
plt.title("Image originale - Espace RGB")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(grey, cmap="gray")
plt.title("Image originale - Espace GRAY")
plt.axis("off")

# Accès pixel (Indexation à 0 en Python !)
pix = flowers[275, 317, :]
print(pix)

plt.imsave(f"Res_test/1_01_originale.png", flowers, cmap="gray")
plt.imsave(f"Res_test/1_02_gray.png", grey, cmap="gray")
plt.show()