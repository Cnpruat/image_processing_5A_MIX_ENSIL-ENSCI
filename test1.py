import cv2
import numpy as np
import matplotlib.pyplot as plt

flowers = cv2.imread("./Image_files/flowers8.png")
# Conversion BGR -> RGB pour l’affichage correct
flowers = cv2.cvtColor(flowers, cv2.COLOR_BGR2RGB)
print(flowers.shape, flowers.dtype)
grey = cv2.cvtColor(flowers, cv2.COLOR_RGB2GRAY)
plt.imshow(grey, cmap="gray")
plt.show()
# Accès pixel (Indexation à 0 en Python !)
pix = flowers[275, 317, :]
print(pix)