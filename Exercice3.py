#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ------------- Cas 1 -------------
yellow = cv2.imread("./Image_files/yellowtargets.png")
yellow_RGB = cv2.cvtColor(yellow, cv2.COLOR_BGR2RGB)
yellow_HSV = cv2.cvtColor(yellow, cv2.COLOR_BGR2HSV)

yellow_RGB_double = yellow_RGB.astype(float) / 255.0
yellow_HSV_double = yellow_HSV.astype(float) / 255.0

plt.figure(figsize=(15,8))
plt.subplot(3,5,1)
plt.imshow(yellow_RGB_double)
plt.title("Espace RGB")
plt.axis("off")

plt.subplot(3,5,6)
plt.imshow(yellow_HSV_double)
plt.title("Espace HSV")
plt.axis("off")

# ------------- Question 1 -------------
# Seuillage
# Calcul de l’intensité (Somme R+G+B)
S = np.sum(yellow_HSV_double, axis=2)
S[S==0] = 1.0 # Eviter division par zero

# Choix couleur sur laquelle on travaille (Attention: HSV -> H->0, S->1, V->2)
h = yellow_HSV_double[:,:,0] / S
s = yellow_HSV_double[:,:,1] / S
v = yellow_HSV_double[:,:,2] / S

mask_v = ((v > 0.55) & (h < 0.1))

# On étend le masque en 3D
mask_3dv = np.stack([mask_v]*3, axis=2) 

# On applique le masque à l'image originale
resultat_v = yellow_RGB_double * mask_3dv

plt.subplot(3,5,11)
plt.imshow(resultat_v)
plt.title("Résultat après application du 1er masque ")
plt.axis("off")

# ------------- Question 2 -------------
# Elément structurant pour ouverture et fermeture
# (5,5) / (7,7) / (9,9)
kernel = np.ones((7,7), np.uint8)

# On repasse en entier (0 - 255)
mask_v_uint8 = (mask_v * 255).astype(np.uint8)

# OUVERTURE = Erosion -> Dilatation (Pour supprimer le bruit blanc autour) 
mask_ouverture = cv2.morphologyEx(mask_v_uint8, cv2.MORPH_OPEN, kernel)

# FERMETURE = Dilatation -> Erosion (Pour combler les trous noirs dedans)
mask_fermeture = cv2.morphologyEx(mask_v_uint8, cv2.MORPH_CLOSE, kernel)

# Combinaisons d'ouverture et fermeture
mask_ouverture_fermeture = cv2.morphologyEx(mask_ouverture, cv2.MORPH_CLOSE, kernel)
mask_fermeture_ouverture = cv2.morphologyEx(mask_fermeture, cv2.MORPH_CLOSE, kernel)

plt.subplot(3,5,2)
plt.imshow(mask_v_uint8, cmap='gray')
plt.title("Masque après seuillage")
plt.axis("off")

plt.subplot(3,5,7)
plt.imshow(mask_ouverture, cmap='gray')
plt.title("Masque après Ouverture")
plt.axis("off")

plt.subplot(3,5,12)
plt.imshow(mask_ouverture_fermeture, cmap='gray')
plt.title("Masque après Fermeture")
plt.axis("off")

# On étend le masque en 3D
mask_3d_ouverture = (np.stack([mask_ouverture]*3, axis=2)).astype(float) / 255.0 
mask_3d_fermeture = (np.stack([mask_fermeture]*3, axis=2)).astype(float) / 255.0 
mask_3d_ouverture_fermeture = (np.stack([mask_ouverture_fermeture]*3, axis=2)).astype(float) / 255.0 
mask_3d_fermeture_ouverture = (np.stack([mask_fermeture_ouverture]*3, axis=2)).astype(float) / 255.0

# On applique le masque à l'image originale
resultat_ouverture = yellow_RGB_double * mask_3d_ouverture
resultat_fermeture = yellow_RGB_double * mask_3d_fermeture
resultat_ouverture_fermeture = yellow_RGB_double * mask_3d_ouverture_fermeture
resultat_fermeture_ouverture = yellow_RGB_double * mask_3d_fermeture_ouverture

plt.subplot(3,5,3)
plt.imshow(resultat_ouverture, cmap='gray')
plt.title("Résultat après ouverture")
plt.axis("off")

plt.subplot(3,5,4)
plt.imshow(resultat_fermeture, cmap='gray')
plt.title("Résultat après fermeture")
plt.axis("off")

plt.subplot(3,5,8)
plt.imshow(resultat_ouverture_fermeture, cmap='gray')
plt.title("Ouverture puis fermeture")
plt.axis("off")

plt.subplot(3,5,9)
plt.imshow(resultat_fermeture_ouverture, cmap='gray')
plt.title("Fermeture puis ouverture")
plt.axis("off")


# ------------- Question 3 -------------
mask_erosion = cv2.erode(mask_ouverture_fermeture, kernel, iterations=1)
mask_dilatation = cv2.dilate(mask_ouverture_fermeture, kernel, iterations=1)

mask_contour_int = mask_ouverture_fermeture - mask_erosion
mask_contour_ext = mask_dilatation - mask_ouverture_fermeture
mask_contour_grad_morphologique = mask_dilatation - mask_erosion

# On étend le masque en 3D
mask_contour_int_3D = (np.stack([mask_contour_int]*3, axis=2)).astype(float) / 255.0 
mask_contour_ext_3D = (np.stack([mask_contour_ext]*3, axis=2)).astype(float) / 255.0 
mask_contour_grad_morphologique_3D = (np.stack([mask_contour_grad_morphologique]*3, axis=2)).astype(float) / 255.0 

# On applique le masque à l'image originale
resultat__int = yellow_RGB_double * mask_contour_int_3D
resultat_ext = yellow_RGB_double * mask_contour_ext_3D
resultat_grad_morphologique = yellow_RGB_double * mask_contour_grad_morphologique_3D

plt.subplot(3,5,5)
plt.imshow(resultat__int, cmap='gray')
plt.title("Contour int")
plt.axis("off")

plt.subplot(3,5,10)
plt.imshow(resultat_ext, cmap='gray')
plt.title("Contour ext")
plt.axis("off")

plt.subplot(3,5,15)
plt.imshow(resultat_grad_morphologique, cmap='gray')
plt.title("Gradient morphologique")
plt.axis("off")

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Organisation de la fenêtre
plt.show()