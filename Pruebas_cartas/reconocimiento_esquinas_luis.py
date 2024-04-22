import cv2
import numpy as np

# Cargar la imagen y el template
img = cv2.imread('cartas.png', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('10-treboles.jpg', cv2.IMREAD_GRAYSCALE)

# Inicializar SIFT
sift = cv2.SIFT_create()

# Encontrar los keypoints y descriptores con SIFT
kp1, des1 = sift.detectAndCompute(img, None)
kp2, des2 = sift.detectAndCompute(template, None)

# Inicializar el matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Realizar el matching
matches = bf.match(des1, des2)

# Ordenar los matches por distancia
matches = sorted(matches, key=lambda x: x.distance)

# Dibujar los primeros 10 matches
img3 = cv2.drawMatches(img, kp1, template, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Mostrar la imagen
cv2.imshow('Matches', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()