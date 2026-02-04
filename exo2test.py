import cv2
import matplotlib.pyplot as plt

#---------------- CAS 1 ----------------
# 1. Chargement de l'image
img_bgr = cv2.imread('./Image_files/cas_1.png')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 2. Application du filtre Médian
# Le deuxième paramètre est la taille du noyau (ksize).
# Il doit être impair (3, 5, 7, etc.).
# Vu la densité du bruit sur ton image, 3 risque d'être juste, essayons 5.
image_median = cv2.medianBlur(img_rgb, 5)

# (Optionnel) Pour comparer : Essai avec un filtre Gaussien
image_gaussian = cv2.GaussianBlur(img_rgb, (5, 5), 0)

# 3. Affichage
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Originale (Bruit Poivre & Sel)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(image_gaussian)
plt.title("Filtre Gaussien (Inefficace ici)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(image_median)
plt.title("Filtre Médian (k=5)")
plt.axis("off")


#---------------- CAS 2 ----------------
# 1. Chargement
img_bgr = cv2.imread('./Image_files/cas_2.png')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 2. Solution Classique : Filtre Gaussien
# On utilise un noyau (5,5). Si l'image est très bruitée, on peut monter à (7,7) ou (9,9)
# Mais attention, plus c'est grand, plus c'est flou.
image_gaussian = cv2.GaussianBlur(img_rgb, (5, 5), 0)

# 3. Solution "Expert" : Filtre Bilatéral
# Paramètres : (src, d, sigmaColor, sigmaSpace)
# d=9 : Diamètre du voisinage (un peu comme ksize)
# sigmaColor=75 : À quel point on mélange les couleurs (plus c'est haut, plus ça lisse les zones unies)
# sigmaSpace=75 : À quel point on mélange les pixels éloignés géographiquement
image_bilateral = cv2.bilateralFilter(img_rgb, 9, 75, 75)

# 4. Affichage
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Originale (Bruit Gaussien/Grain)")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(image_gaussian)
plt.title("Filtre Gaussien (5x5)\nEfficace mais flou")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(image_bilateral)
plt.title("Filtre Bilatéral\nLisse le grain, garde les bords")
plt.axis("off")


#---------------- CAS 3 ----------------
# 1. Chargement de l'image (en niveaux de gris pour commencer simple)
# Si tu veux le faire en couleur, il faut convertir en HSV, étirer le canal V, et reconvertir.
# Pour l'exercice, supposons qu'on travaille en noir et blanc ou sur le canal de luminosité.
img = cv2.imread('./Image_files/cas_3.png') # 0 pour charger directement en gris
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- ETAPE 1 : Le Recadrage Dynamique (Stretching) ---
# Méthode manuelle (comme dans le cours) ou via OpenCV
# On utilise cv2.normalize qui fait exactement la formule du cours de manière optimisée
img_etire = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# --- ETAPE 2 : Comparaison des Histogrammes ---
plt.figure(figsize=(12, 8))

# Image Originale
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title("Originale (Faible contraste)")
plt.axis("off")

# Histogramme Original
plt.subplot(2, 2, 2)
plt.hist(img.ravel(), 256, [0, 256], color='gray')
plt.title("Histogramme Original (Concentré)")

# Image Étirée
plt.subplot(2, 2, 3)
plt.imshow(img_etire, cmap='gray', vmin=0, vmax=255)
plt.title("Après Recadrage Dynamique")
plt.axis("off")

# Histogramme Étiré
plt.subplot(2, 2, 4)
plt.hist(img_etire.ravel(), 256, [0, 256], color='gray')
plt.title("Histogramme Étiré (Tout l'espace utilisé)")


plt.tight_layout()
plt.show()