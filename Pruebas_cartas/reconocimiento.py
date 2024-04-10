# import cv2

# # Cargar la imagen
# img = cv2.imread('10-treboles.jpg')

# # Crear el detector ORB con parámetros ajustados
# orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.1)

# # Detectar las características
# keypoints, descriptors = orb.detectAndCompute(img, None)

# # Dibujar las características detectadas en la imagen
# img_with_keypoints = cv2.drawKeypoints(img, keypoints, None)

# # Mostrar la imagen
# cv2.imshow('Características', img_with_keypoints)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # Cargar la imagen y el template
# img = cv2.imread('cartas.png', cv2.IMREAD_GRAYSCALE)
# template = cv2.imread('10-treboles.jpg', cv2.IMREAD_GRAYSCALE)

# # Obtener las dimensiones del template
# w, h = template.shape[::-1]

# # Realizar el template matching
# res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
# threshold = 0.8
# loc = np.where(res >= threshold)

# # Dibujar un rectángulo alrededor de la zona detectada
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

# # Mostrar la imagen
# cv2.imshow('Detected', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np

# Cargar la imagen y el template
img = cv2.imread('cartas.png', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('3_rombos.jpg', cv2.IMREAD_GRAYSCALE)

# Inicializar SIFT
sift = cv2.SIFT_create()

# Encontrar los keypoints y descriptores con SIFT
kp1, des1 = sift.detectAndCompute(img, None)
kp2, des2 = sift.detectAndCompute(template, None)

# Inicializar el matcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Aplicar la prueba de proporción de Lowe
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

# Si hay suficientes buenos matches, encontrar la homografía
if len(good) > 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = template.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img = cv2.polylines(img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

# Mostrar la imagen
cv2.imshow('Detected', img)
cv2.waitKey(0)
cv2.destroyAllWindows()