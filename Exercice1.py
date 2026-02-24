#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

shapes = cv2.imread("./Image_files/shapes.png")
shapes = cv2.cvtColor(shapes, cv2.COLOR_BGR2RGB)

# Normalisation en flottant (0.0 - 1.0)
shapes_double = shapes.astype(float) / 255.0

plt.figure(figsize=(15,8))
plt.suptitle("Exercice préliminaire - Traitements sur l'image shapes.png", weight='bold')
# L’image
plt.subplot(3,4,1)
plt.imshow(shapes_double)
plt.title("Image originale")
plt.axis("off")

# ------------- Question 1 -------------
# Histogramme
hist_vals, bins = np.histogram(shapes_double.ravel(), bins=256, range=(0,1))
plt.subplot(3,4,9)
plt.plot(hist_vals)
plt.title("Histogramme de l'image")

# ------------- Question 2 -------------
# Seuillage
# Calcul de l’intensité (Somme R+G+B)
S = np.sum(shapes_double, axis=2)
S[S==0] = 1.0 # Eviter division par zero

# Choix couleur sur laquelle on travaille (Attention: RGB -> R->0, G->1, B->2)
r = shapes_double[:,:,0] / S
b = shapes_double[:,:,2] / S

maskBF = (b > 0.5) # Que bleu foncé
maskBC = ((b<0.5) & (b > 0.4) & (r < 0.4)) # Que bleu ciel
maskB = ((b > 0.4) & (r < 0.4)) # Bleu foncé ET ciel

# Affichage des masques crées
plt.subplot(3,4,2)
plt.imshow(maskB, cmap="gray")
plt.title("Masque bleu")
plt.axis("off")
plt.subplot(3,4,6)
plt.imshow(maskBF, cmap="gray")
plt.title("Masque bleu foncé")
plt.axis("off")
plt.subplot(3,4,10)
plt.imshow(maskBC, cmap="gray")
plt.title("Masque bleu ciel")
plt.axis("off")

# On étend les masques en 3D
mask_3dF = np.stack([maskBF]*3, axis=2) 
mask_3dC = np.stack([maskBC]*3, axis=2) 
mask_3d = np.stack([maskB]*3, axis=2) 

# On applique les masques à l'image originale
resultatF = shapes_double * mask_3dF
resultatC = shapes_double * mask_3dC
resultat = shapes_double * mask_3d

# Affichage des résultats
plt.subplot(3,4,3)
plt.imshow(resultat)
plt.title("Sélection - Bleus")
plt.axis("off")
plt.subplot(3,4,7)
plt.imshow(resultatF)
plt.title("Sélection - Foncé")
plt.axis("off")
plt.subplot(3,4,11)
plt.imshow(resultatC)
plt.title("Sélection - Ciel")
plt.axis("off")

# ------------- Question 3 -------------
# On repasse en entier (0 - 255)
mask_uint8 = (maskB * 255).astype(np.uint8)

# On localise les contours sur notre masque
contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Masque vide
mask_cercle = np.zeros_like(mask_uint8)

i = 1 # Compteur

# On parcourt tous les contours trouvés par la fct findContours
for cnt in contours:
    # Calculs de l'aire et du périmètre
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    # Permet d'ignorer les artefacts de quelques pixels qui ne sont pas de "vraies" formes
    if area < 1: 
        continue

    # Évite la division par zéro
    if perimeter == 0: 
        continue 
        
    # Calcul de la circularité : 4 * pi * Aire / Perimetre^2
    circ = 4 * np.pi * area / (perimeter ** 2)
    
    print("Objet n°", i, "d'aire = ",area, "et de circularité = ",f"{circ:.2f}")
    
    # Si la circularité est supérieure à 0.85 on considère que c'est un cercle
    if circ > 0.85:
        # On dessine ce contour en blanc (255) sur notre masque final
        cv2.drawContours(mask_cercle, [cnt], -1, 255, -1)
    i+=1

# Application du masque cercle à l'image originale pour visualisation
mask_cercle_3d = np.stack([mask_cercle/255]*3, axis=2) # Extension du masque en 3D
resultat_final = shapes_double * mask_cercle_3d        # Application à notre image originale

plt.subplot(3,4,4)
plt.imshow(resultat_final)
plt.title("Résultat - Uniquement cercle bleu")
plt.axis("off")

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Organisation de la fenêtre

plt.imsave(f"Res_Ex1/01_originale.png", shapes_double)

plt.imsave(f"Res_Ex1/02_masque_bleu.png", maskB, cmap="gray")
plt.imsave(f"Res_Ex1/03_masque_bleu_fonce.png", maskBF, cmap="gray")
plt.imsave(f"Res_Ex1/04_masque_bleu_ciel.png", maskBC, cmap="gray")

plt.imsave(f"Res_Ex1/05_selection_bleus.png", resultat)
plt.imsave(f"Res_Ex1/06_selection_fonce.png", resultatF)
plt.imsave(f"Res_Ex1/07_selection_ciel.png", resultatC)

plt.imsave(f"Res_Ex1/08_resultat_cercle_bleu.png", resultat_final)
plt.show()
# %%