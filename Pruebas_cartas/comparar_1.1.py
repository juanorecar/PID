import cv2
import numpy as np
import os
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
def determinar_carta(density_map, densidades_referencia):
    mejor_carta = None
    mejor_diferencia = float('inf')
    
    for carta, densidad_referencia in densidades_referencia.items():
        diferencia = np.abs(density_map.sum() - densidad_referencia.sum())
        if diferencia < mejor_diferencia:
            mejor_carta = carta
            mejor_diferencia = diferencia
    
    return mejor_carta

# Cargar la imagen y el template
img = cv2.imread('10_treboles.jpg', cv2.IMREAD_GRAYSCALE)

# Inicializar SIFT
sift = cv2.SIFT_create()

# Inicializar el matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Aquí debes definir tus densidades de referencia para cada carta
# Puedes calcular estas densidades de referencia utilizando una imagen de referencia para cada carta
# Por ejemplo, podrías tener un diccionario donde las claves son nombres de cartas y los valores son densidades de referencia
# Aquí estoy utilizando densidades de referencia ficticias solo para propósitos de demostración

    

# Diccionario de densidades de referencia ficticias para cada carta
densidades_referencia = {
    "10_corazones": np.array([[0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 2.0, 0.0], [3.0, 2.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]]),
    "10_picas": np.array([[2.0, 10.0, 5.0, 3.0], [3.0, 10.0, 8.0, 4.0], [1.0, 4.0, 14.0, 3.0], [0.0, 12.0, 15.0, 3.0]]),
    "10_rombos": np.array([[2.0, 3.0, 4.0, 1.0], [0.0, 5.0, 9.0, 3.0], [0.0, 0.0, 4.0, 0.0], [0.0, 0.0, 3.0, 1.0]]),
    "10_treboles": np.array([[33.0, 17.0, 12.0, 7.0], [7.0, 59.0, 59.0, 8.0], [6.0, 37.0, 48.0, 6.0], [3.0, 14.0, 14.0, 29.0]]),
    "2_corazones": np.array([[0.0, 2.0, 4.0, 0.0], [0.0, 7.0, 7.0, 0.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 3.0, 0.0]]),
    "2_picas": np.array([[2.0, 3.0, 3.0, 1.0], [0.0, 3.0, 4.0, 4.0], [0.0, 2.0, 7.0, 0.0], [0.0, 0.0, 4.0, 0.0]]),
    "2_rombos": np.array([[0.0, 3.0, 2.0, 0.0], [0.0, 5.0, 9.0, 0.0], [0.0, 3.0, 11.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
    "2_treboles": np.array([[0.0, 6.0, 0.0, 0.0], [0.0, 5.0, 6.0, 0.0], [0.0, 4.0, 5.0, 0.0], [0.0, 5.0, 5.0, 0.0]]),
    "3_corazones": np.array([[3.0, 5.0, 7.0, 0.0], [3.0, 2.0, 2.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]]),
    "3_picas": np.array([[0.0, 8.0, 10.0, 0.0], [0.0, 4.0, 2.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
    "3_rombos": np.array([[0.0, 6.0, 8.0, 0.0], [2.0, 6.0, 4.0, 0.0], [1.0, 7.0, 7.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
    "3_treboles": np.array([[0.0, 10.0, 14.0, 0.0], [0.0, 6.0, 6.0, 0.0], [0.0, 6.0, 6.0, 0.0], [0.0, 8.0, 9.0, 0.0]]),
    "4_corazones": np.array([[0.0, 0.0, 7.0, 1.0], [0.0, 6.0, 12.0, 13.0], [1.0, 1.0, 7.0, 0.0], [0.0, 1.0, 1.0, 0.0]]),
    "4_picas": np.array([[0.0, 0.0, 5.0, 0.0], [0.0, 2.0, 9.0, 11.0], [0.0, 3.0, 1.0, 1.0], [0.0, 3.0, 6.0, 0.0]]),
    "4_rombos": np.array([[0.0, 1.0, 1.0, 0.0], [5.0, 4.0, 9.0, 3.0], [2.0, 5.0, 5.0, 0.0], [1.0, 4.0, 1.0, 0.0]]),
    "4_treboles": np.array([[0.0, 0.0, 6.0, 0.0], [0.0, 5.0, 10.0, 6.0], [0.0, 4.0, 3.0, 0.0], [0.0, 2.0, 3.0, 0.0]]),
    "5_corazones": np.array([[0.0, 4.0, 3.0, 0.0], [2.0, 0.0, 1.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 2.0, 2.0, 0.0]]),
    "5_picas": np.array([[0.0, 9.0, 4.0, 0.0], [2.0, 10.0, 5.0, 1.0], [0.0, 6.0, 0.0, 0.0], [0.0, 12.0, 6.0, 0.0]]),
    "5_rombos": np.array([[4.0, 5.0, 4.0, 0.0], [7.0, 2.0, 1.0, 1.0], [1.0, 9.0, 1.0, 0.0], [5.0, 4.0, 2.0, 0.0]]),
    "5_treboles": np.array([[0.0, 6.0, 3.0, 1.0], [3.0, 0.0, 0.0, 0.0], [0.0, 2.0, 2.0, 0.0], [2.0, 2.0, 4.0, 2.0]]),
    "6_corazones": np.array([[0.0, 5.0, 3.0, 0.0], [3.0, 2.0, 2.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 2.0, 1.0]]),
    "6_picas": np.array([[0.0, 1.0, 2.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0]]),
    "6_rombos": np.array([[0.0, 4.0, 2.0, 0.0], [0.0, 2.0, 2.0, 1.0], [0.0, 4.0, 7.0, 0.0], [0.0, 0.0, 3.0, 0.0]]),
    "6_treboles": np.array([[0.0, 7.0, 9.0, 2.0], [2.0, 6.0, 4.0, 7.0], [1.0, 4.0, 9.0, 0.0], [3.0, 2.0, 7.0, 2.0]]),
    "7_corazones": np.array([[2.0, 5.0, 9.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 11.0, 1.0, 0.0], [0.0, 8.0, 2.0, 0.0]]),
    "7_picas": np.array([[0.0, 1.0, 3.0, 2.0], [0.0, 2.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0], [0.0, 3.0, 1.0, 1.0]]),
    "7_rombos": np.array([[2.0, 1.0, 2.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 3.0, 4.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
    "7_treboles": np.array([[2.0, 2.0, 7.0, 2.0], [0.0, 2.0, 1.0, 0.0], [0.0, 3.0, 2.0, 0.0], [2.0, 3.0, 4.0, 0.0]]),
    "8_corazones": np.array([[0.0, 5.0, 0.0, 0.0], [2.0, 2.0, 1.0, 1.0], [3.0, 6.0, 0.0, 1.0], [1.0, 4.0, 1.0, 0.0]]),
    "8_picas": np.array([[4.0, 3.0, 1.0, 1.0], [0.0, 2.0, 9.0, 0.0], [0.0, 4.0, 5.0, 0.0], [1.0, 5.0, 1.0, 0.0]]),
    "8_rombos": np.array([[0.0, 1.0, 3.0, 1.0], [7.0, 5.0, 5.0, 3.0], [0.0, 7.0, 3.0, 2.0], [0.0, 4.0, 2.0, 2.0]]),
    "8_treboles": np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 3.0, 5.0, 0.0], [0.0, 2.0, 3.0, 0.0], [2.0, 6.0, 4.0, 2.0]]),
    "9_corazones": np.array([[0.0, 0.0, 1.0, 0.0], [2.0, 0.0, 6.0, 1.0], [0.0, 3.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
    "9_picas": np.array([[1.0, 3.0, 2.0, 0.0], [1.0, 2.0, 4.0, 0.0], [0.0, 2.0, 2.0, 0.0], [0.0, 0.0, 4.0, 1.0]]),
    "9_rombos": np.array([[2.0, 0.0, 3.0, 0.0], [1.0, 2.0, 6.0, 0.0], [0.0, 6.0, 1.0, 2.0], [0.0, 0.0, 3.0, 0.0]]),
    "9_treboles": np.array([[0.0, 1.0, 2.0, 0.0], [0.0, 6.0, 3.0, 3.0], [0.0, 0.0, 6.0, 0.0], [1.0, 5.0, 5.0, 2.0]]),
    "as_de_rombos": np.array([[0.0, 9.0, 11.0, 0.0], [3.0, 42.0, 39.0, 0.0], [0.0, 6.0, 4.0, 2.0], [1.0, 9.0, 4.0, 1.0]]),
    "As_corazones": np.array([[0.0, 23.0, 22.0, 0.0], [3.0, 69.0, 51.0, 3.0], [19.0, 9.0, 12.0, 10.0], [2.0, 8.0, 7.0, 0.0]]),
    "as_picas": np.array([[0.0, 7.0, 5.0, 0.0], [4.0, 8.0, 8.0, 3.0], [0.0, 4.0, 1.0, 0.0], [1.0, 3.0, 3.0, 2.0]]),
    "As_treboles": np.array([[0.0, 39.0, 31.0, 0.0], [10.0, 83.0, 81.0, 0.0], [8.0, 15.0, 21.0, 0.0], [1.0, 23.0, 13.0, 1.0]]),
    "cartas": np.array([[22.0, 110.0, 33.0, 9.0], [57.0, 49.0, 12.0, 39.0], [66.0, 15.0, 89.0, 3.0], [40.0, 22.0, 27.0, 54.0]]),
    "J_corazones": np.array([[0.0, 0.0, 6.0, 1.0], [4.0, 3.0, 1.0, 0.0], [1.0, 3.0, 2.0, 0.0], [0.0, 10.0, 2.0, 0.0]]),
    "j_picas": np.array([[0.0, 2.0, 5.0, 4.0], [2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 1.0, 3.0, 0.0]]),
    "j_rombos": np.array([[2.0, 3.0, 3.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 5.0, 6.0, 0.0], [0.0, 0.0, 2.0, 0.0]]),
    "j_treboles": np.array([[2.0, 6.0, 7.0, 1.0], [1.0, 2.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [2.0, 2.0, 3.0, 0.0]]),
    "K_corazones": np.array([[2.0, 4.0, 3.0, 2.0], [1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [2.0, 4.0, 3.0, 1.0]]),
    "k_picas": np.array([[3.0, 0.0, 3.0, 2.0], [1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [3.0, 1.0, 2.0, 1.0]]),
    "k_rombos": np.array([[1.0, 2.0, 2.0, 2.0], [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 2.0, 1.0, 1.0]]),
    "K_treboles": np.array([[4.0, 5.0, 7.0, 0.0], [2.0, 0.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0], [2.0, 3.0, 2.0, 2.0]]),
    "Q_corazones": np.array([[0.0, 2.0, 3.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
    "Q_picas": np.array([[2.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
    "Q_rombos": np.array([[1.0, 0.0, 2.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
    "Q_treboles": np.array([[0.0, 2.0, 2.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
}

# Guardar el diccionario de densidades de referencia en un archivo
np.save('densidades_referencia.npy', densidades_referencia)

    # Agrega más cartas con sus densidades de referencia


# Encontrar los keypoints y descriptores con SIFT para la imagen de entrada
kp1, des1 = sift.detectAndCompute(img, None)

# Inicializar el matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Inicializar una bandera para controlar si se ha impreso una respuesta
respuesta_impresa = False

# Iterar sobre las densidades de referencia y determinar la carta más probable
for carta, densidad_referencia in densidades_referencia.items():
    # Cargar el template de la carta actual
    template_path = '{}.png'.format(carta)
    if not os.path.exists(template_path):
        template_path = '{}.jpg'.format(carta)

    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    if template is None:
        print(f"Error: No se pudo cargar el template {carta}")
        continue  # Salta a la siguiente iteración del bucle

    # Continuar con el procesamiento solo si la imagen se cargó correctamente
    kp2, des2 = sift.detectAndCompute(template, None)

    # Realizar el matching con el descriptor BF
    matches = bf.match(des1, des2)

    # Ordenar los matches por distancia
    matches = sorted(matches, key=lambda x: x.distance)

    # Seleccionar las mejores coincidencias (puedes ajustar este valor según tu necesidad)
    good_matches = matches[:10]

    # Calcular la densidad de puntos coincidentes
    density_map = calcular_densidad(good_matches, kp1, img.shape[1], img.shape[0], grid_size=4)
    
    # Determinar la carta más probable
    mejor_carta = determinar_carta(density_map, densidades_referencia)
    
    # Imprimir la carta detectada
    if mejor_carta:
        print(f"La carta detectada es: {mejor_carta}")
        respuesta_impresa = True
        break

# Si no se ha impreso ninguna respuesta, mostrar un mensaje de error
if not respuesta_impresa:
    print("No se pudo detectar ninguna carta.")

cv2.waitKey(0)
cv2.destroyAllWindows()