import cv2
import numpy as np
import os

# Inicializar SIFT
sift = cv2.SIFT_create()

# Función para calcular las densidades de los keypoints en una cuadrícula
def calcular_densidad(kp, img_width, img_height, grid_size):
    density_map = np.zeros((grid_size, grid_size))
    cell_width = img_width // grid_size
    cell_height = img_height // grid_size
    
    for point in kp:
        x, y = point.pt
        col = int(x / cell_width)
        row = int(y / cell_height)
        density_map[row, col] += 1
        
    return density_map

# Función para cargar las imágenes de referencia y calcular las densidades de referencia
def calcular_densidades_referencia():
    # Diccionario para almacenar las densidades de referencia
    densidades_referencia = {}

    # Obtener la lista de archivos en el directorio actual
    archivos = os.listdir('.')
    
    # Iterar sobre cada archivo en el directorio
    for archivo in archivos:
        # Verificar si el archivo es una imagen JPG o PNG
        if archivo.endswith('.jpg') or archivo.endswith('.png'):
            # Extraer el nombre de la carta del archivo
            nombre_carta = os.path.splitext(archivo)[0]
            # Cargar la imagen de referencia de la carta
            carta_img = cv2.imread(archivo, cv2.IMREAD_GRAYSCALE)
            # Encontrar los keypoints y descriptores con SIFT
            kp, des = sift.detectAndCompute(carta_img, None)
            # Calcular la densidad de puntos coincidentes en una cuadrícula 4x4
            density_map = calcular_densidad(kp, carta_img.shape[1], carta_img.shape[0], grid_size=4)
            # Almacenar la densidad de referencia en el diccionario
            densidades_referencia[nombre_carta] = density_map

    return densidades_referencia

# Calcular las densidades de referencia
densidades_referencia = calcular_densidades_referencia()

# Imprimir las densidades de referencia
for carta, density_map in densidades_referencia.items():
    print(f'Densidad de referencia para {carta}:')
    for row in density_map:
        print("[", ", ".join(map(str, row)), "]")
    print('\n')
