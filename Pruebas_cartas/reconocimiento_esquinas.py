import cv2
import numpy as np

# Cargar la imagen y el template
img = cv2.imread('cartas.png', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('10_treboles.jpg', cv2.IMREAD_GRAYSCALE)

# Inicializar SIFT
sift = cv2.SIFT_create()

# Encontrar los keypoints y descriptores con SIFT
kp1, des1 = sift.detectAndCompute(img, None)
kp2, des2 = sift.detectAndCompute(template, None)

# Obtener las coordenadas de los keypoints
keypoint_coords = np.array([kp.pt for kp in kp1], dtype=np.float32)

# Definir el número de clusters
num_clusters = 5

# Aplicar el algoritmo de K-Means para agrupar los keypoints
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(keypoint_coords, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convertir los centros de los clusters a enteros
centers = np.int0(centers)

# Encontrar el centroide más cercano al centro de la imagen
center_of_image = (img.shape[1] // 2, img.shape[0] // 2)
closest_center_idx = np.argmin(np.linalg.norm(centers - center_of_image, axis=1))

# Obtener las coordenadas del centroide más cercano
max_cluster_center = centers[closest_center_idx]

# Dibujar un círculo en la imagen original para resaltar la zona de mayor densidad
img_with_circle = cv2.circle(img.copy(), (max_cluster_center[0], max_cluster_center[1]), 50, (255, 0, 0), 3)

# Mostrar la imagen con el círculo
cv2.imshow('Image with Max Density Area', img_with_circle)
cv2.waitKey(0)
cv2.destroyAllWindows()
