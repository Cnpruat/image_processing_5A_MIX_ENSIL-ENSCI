#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(15,8))
plt.suptitle("Exercice 2 - Traitements sur 3 cas avec problèmes de bruit et/ou de constraste", weight='bold')

# ------------- Cas 1 -------------
cas1 = cv2.imread("./Image_files/cas_1.png")
cas1 = cv2.cvtColor(cas1, cv2.COLOR_BGR2RGB)

cas1_double = cas1.astype(float) / 255.0

# Image originale
plt.subplot(3,4,1)
plt.imshow(cas1_double)
plt.title("Cas 1")
plt.axis("off")

# On identifie un bruit poivre et sel --> filtre médian efficace
# Application du filtre médian (Valeur : 3, 5, 7,...)
cas_1_median = cv2.medianBlur(cas1, 5)
plt.subplot(3,4,5)
plt.imshow(cas_1_median)
plt.title("Filtre médian")
plt.axis("off")

# Création du noyau pour réhausser les contours
kernel_sharpening = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])

# Application du filtre réhausseur de contours
cas_1_sharp = cv2.filter2D(cas_1_median, -1, kernel_sharpening)

plt.subplot(3,4,9)
plt.imshow(cas_1_sharp)
plt.title("Filtre réhausseur de contours")
plt.axis("off")

# ------------- Cas 2 -------------
cas2 = cv2.imread("./Image_files/cas_2.png")
cas2 = cv2.cvtColor(cas2, cv2.COLOR_BGR2RGB)

cas2_double = cas2.astype(float) / 255.0

plt.subplot(3,4,2)
plt.imshow(cas2_double)
plt.title("Cas 2")
plt.axis("off")

# On identifie un bruit gaussien --> filtre gaussien efficace
cas2_gaussian = cv2.GaussianBlur(cas2, (5,5), 0)
plt.subplot(3,4,6)
plt.imshow(cas2_gaussian)
plt.title("Filtre Gaussien")
plt.axis("off")

# On test aussi le filtre bilatéral qui fait la même chose que le 
# gaussien mais en prenant en compte les contours pour moins déformer
cas2_bilateral = cv2.bilateralFilter(cas2, 9, 75, 75)
plt.subplot(3,4,10)
plt.imshow(cas2_bilateral)
plt.title("Filtre bilatéral (Gaussien + préservation des bords)")
plt.axis("off")

# ------------- Cas 3 -------------
cas3 = cv2.imread("./Image_files/cas_3.png")
cas3 = cv2.cvtColor(cas3, cv2.COLOR_BGR2RGB)

# Normalisation en flottant (0.0 - 1.0)
cas3_double = cas3.astype(float) / 255.0

# Image originale
plt.subplot(3,4,3)
plt.imshow(cas3_double)
plt.title("Cas 3")
plt.axis("off")

hist_vals, bins = np.histogram(cas3_double.ravel(), bins=256, range=(0,1))
plt.subplot(3,4,7)
plt.plot(hist_vals)
plt.title("Histogramme - cas 3")

# On remarque que l'image n'est pas assez constrastée sur l'histogramme
# On normalise son histogramme pour réhausser le contraste
cas3_normal = cv2.normalize(cas3, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
plt.subplot(3,4,4)
plt.imshow(cas3_normal)
plt.title("Cas 3 - Normalisée")
plt.axis("off")

# Normalisation en flottant (0.0 - 1.0)
cas3_normal_double = cas3_normal.astype(float) / 255.0

# Histogramme normalisé
hist_vals, bins = np.histogram(cas3_normal_double.ravel(), bins=256, range=(0,1))
plt.subplot(3,4,8)
plt.plot(hist_vals)
plt.title("Histogramme - cas 3 normalisé")

# On observe le même bruit gaussien que le cas 2 donc on applique un filtre bilatéral
cas3_normal_bilateral = cv2.bilateralFilter(cas3_normal, 9, 75, 75)
plt.subplot(3,4,11)
plt.imshow(cas3_normal_bilateral)
plt.title("Cas 3 - Normalisée puis filtre bilatéral")
plt.axis("off")

# Même chose mais on filtre avant de normaliser
cas3_bilateral = cv2.bilateralFilter(cas3, 9, 75, 75)
cas3_bilateral_normal = cv2.normalize(cas3_bilateral, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
plt.subplot(3,4,12)
plt.imshow(cas3_bilateral_normal)
plt.title("Cas 3 - Filtre bilatéral puis normalisée")
plt.axis("off")

# Enregistrement des images
plt.imsave(f"Res_Ex2/01_originale_cas1.png", cas1_double, cmap="gray")
plt.imsave(f"Res_Ex2/02_filre_median_cas1.png", cas_1_median, cmap="gray")
plt.imsave(f"Res_Ex2/03_contour_durcis_cas2.png", cas_1_sharp, cmap="gray")

plt.imsave(f"Res_Ex2/04_originale_cas2.png", cas2_double, cmap="gray")
plt.imsave(f"Res_Ex2/05_filtre_gaussien_cas2.png", cas2_gaussian, cmap="gray")
plt.imsave(f"Res_Ex2/06_filtre_bilatéral_cas2.png", cas2_bilateral, cmap="gray")

plt.imsave(f"Res_Ex2/07_originale_cas3.png", cas3_double, cmap="gray")
plt.imsave(f"Res_Ex2/08_normalisation_cas3.png", cas3_normal, cmap="gray")
plt.imsave(f"Res_Ex2/09_normalisation_bilateral_cas3.png", cas3_normal_bilateral, cmap="gray")
plt.imsave(f"Res_Ex2/10_bilateral_normalisation_cas3.png", cas3_bilateral_normal, cmap="gray")

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Organisation de la fenêtre
plt.show()