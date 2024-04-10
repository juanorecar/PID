
import cv2
import os

def calcular_densidad_puntos(imagen):
    # Inicializar el detector ORB
    orb = cv2.ORB_create()
    # Detectar puntos clave y calcular descriptores
    keypoints, _ = orb.detectAndCompute(imagen, None)
    # El número de puntos clave detectados es una medida de densidad de puntos
    return len(keypoints)

def comparar_imagenes(imagen1, imagen2):
    densidad_puntos_imagen1 = calcular_densidad_puntos(imagen1)
    densidad_puntos_imagen2 = calcular_densidad_puntos(imagen2)
    return abs(densidad_puntos_imagen1 - densidad_puntos_imagen2)

def encontrar_carta_similar(imagen_a_comparar, carpeta_cartas):
    imagenes_cartas = []
    for filename in os.listdir(carpeta_cartas):
        path = os.path.join(carpeta_cartas, filename)
        
        imagen_carta = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        imagenes_cartas.append(imagen_carta)
    
    mejor_coincidencia = None
    menor_diferencia = float('inf')
    for carta in imagenes_cartas:
        diferencia = comparar_imagenes(imagen_a_comparar, carta)
        if diferencia < menor_diferencia:
            mejor_coincidencia = carta
            menor_diferencia = diferencia
    
    return mejor_coincidencia


imagen_a_comparar = cv2.imread('pi.png', cv2.IMREAD_GRAYSCALE)
carpeta_cartas = r'C:\Users\Juan\Desktop\Cartas_BD'
carta_similar = encontrar_carta_similar(imagen_a_comparar, carpeta_cartas)

cv2.imshow('Carta Similar', carta_similar)
cv2.waitKey(0)
cv2.destroyAllWindows()