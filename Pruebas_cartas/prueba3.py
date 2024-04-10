import cv2
import numpy as np

# Cargar la imagen de la carta y la imagen de la escena
carta_img = cv2.imread('3_picas.jpg', 0)  # Asegúrate de cambiar 'carta.png' al nombre de tu imagen de carta
escena_img = cv2.imread('4_rombos.jpg', 0)  # Asegúrate de cambiar 'escena.png' al nombre de tu imagen de escena

# Realizar la comparación de plantillas
result = cv2.matchTemplate(escena_img, carta_img, cv2.TM_CCOEFF_NORMED)

# Establecer un umbral para la coincidencia
threshold = 0.8  # Puedes ajustar este valor según tus necesidades

# Encontrar las ubicaciones donde la coincidencia es mayor que el umbral
loc = np.where(result >= threshold)

# Dibujar un cuadro delimitador alrededor de las ubicaciones encontradas
for pt in zip(*loc[::-1]):
    cv2.rectangle(escena_img, pt, (pt[0] + carta_img.shape[1], pt[1] + carta_img.shape[0]), (0, 255, 0), 2)

# Mostrar la imagen con las detecciones
cv2.imshow('Detección de Cartas', escena_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
