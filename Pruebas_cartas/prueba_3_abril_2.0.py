import cv2
import os

def calcular_descriptores(imagen):
    orb = cv2.ORB_create()
    keypoints, descriptores = orb.detectAndCompute(imagen, None)
    return keypoints, descriptores

def encontrar_carta_similar(imagen_a_comparar, carpeta_cartas):
    orb = cv2.ORB_create()
    kp1, des1 = calcular_descriptores(imagen_a_comparar)
    
    mejor_coincidencia = None
    menor_diferencia = float('inf')
    
    for filename in os.listdir(carpeta_cartas):
        path = os.path.join(carpeta_cartas, filename)
        imagen_carta = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        kp2, des2 = calcular_descriptores(imagen_carta)
        
        if des2 is None:
            continue
        
        # Inicializar FLANN matcher
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                           table_number = 6,
                           key_size = 12,
                           multi_probe_level = 1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        
        # Realizar el matching de descriptores
        matches = flann.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        diferencia = len(good_matches)
        
        if diferencia < menor_diferencia:
            mejor_coincidencia = imagen_carta
            menor_diferencia = diferencia
    
    return mejor_coincidencia

# Ejemplo de uso
imagen_a_comparar = cv2.imread('8_rombos.jpg', cv2.IMREAD_GRAYSCALE)
carpeta_cartas = r'C:\Users\Juan\Desktop\Cartas_BD'
carta_similar = encontrar_carta_similar(imagen_a_comparar, carpeta_cartas)

cv2.imshow('Carta Similar', carta_similar)
cv2.waitKey(0)
cv2.destroyAllWindows()