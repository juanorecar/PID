import cv2
import numpy as np

# Función para calcular la densidad de puntos coincidentes en una cuadrícula
def calcular_densidad(matches, kp1, img_width, img_height, grid_size):
    density_map = np.zeros((grid_size, grid_size))
    cell_width = img_width // grid_size
    cell_height = img_height // grid_size
    
    for match in matches:
        x, y = kp1[match.queryIdx].pt
        col = int(x / cell_width)
        row = int(y / cell_height)
        density_map[row, col] += 1
        
    return density_map

# Comparar densidad_map con densidades de referencia para determinar la carta
def determinar_carta(density_map):
    # Aquí debes definir tus densidades de referencia para cada carta
    # Por ejemplo, podrías tener un diccionario donde las claves son nombres de cartas y los valores son densidades de referencia
    densidades_referencia = {
        "10-treboles": 359,
        "J-diamantes": 25,
        # Agrega más cartas con sus densidades de referencia
    }
    
    mejor_carta = None
    mejor_diferencia = float('inf')
    
    for carta, densidad_referencia in densidades_referencia.items():
        diferencia = np.abs(density_map.sum() - densidad_referencia)
        if diferencia < mejor_diferencia:
            mejor_carta = carta
            mejor_diferencia = diferencia
    
    return mejor_carta

# Cargar la imagen y el template
img = cv2.imread('As_corazones.png', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('As_corazones.png', cv2.IMREAD_GRAYSCALE)

# Inicializar SIFT
sift = cv2.SIFT_create()

# Encontrar los keypoints y descriptores con SIFT
kp1, des1 = sift.detectAndCompute(img, None)
kp2, des2 = sift.detectAndCompute(template, None)

# Inicializar el matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Realizar el matching
matches = bf.match(des1, des2)

# Calcular la densidad de puntos coincidentes en una cuadrícula 4x4 (por ejemplo)
grid_size = 4
density_map = calcular_densidad(matches, kp1, img.shape[1], img.shape[0], grid_size)

# Contar cuántos puntos coincidentes se encontraron
num_puntos_coincidentes = len(matches)
print("Número de puntos coincidentes encontrados:", num_puntos_coincidentes)

# Determinar la carta
carta_detectada = determinar_carta(density_map)

print("La carta detectada es:", carta_detectada)

# Dibujar los primeros 10 matches
img3 = cv2.drawMatches(img, kp1, template, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Mostrar la imagen
cv2.imshow('Matches', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
