import cv2
import numpy as np
from keras.models import load_model


def preprocess_image(image):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar suavizado para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Aplicar umbralización para binarizar la imagen
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    return thresh

def find_card_contours(image):
    # Encontrar contornos en la imagen binarizada
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar los contornos para obtener solo aquellos que probablemente sean cartas
    card_contours = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            card_contours.append(contour)
    
    return card_contours

def filter_card_contours(contours, min_area=1000, max_area=50000):
    # Filtrar contornos basados en el área
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area and area < max_area:
            filtered_contours.append(contour)
    
    return filtered_contours

def recognize_card(image, contour, model):
    # Definir las regiones de interés (ROI) para la número y el palo de la carta
    x, y, w, h = cv2.boundingRect(contour)
    roi = image[y:y+h, x:x+w]
    roi = cv2.resize(roi, (100, 100))  # Ajustar el tamaño de la imagen para el modelo
    
    # Normalizar la imagen
    roi = roi.astype('float') / 255.0
    
    # Agregar una dimensión adicional para que tenga la forma correcta para la entrada del modelo
    roi = np.expand_dims(roi, axis=0)
    
    # Realizar la predicción utilizando el modelo
    prediction = model.predict(roi)
    
    # Decodificar la salida del modelo
    number_label = np.argmax(prediction[0][:13])  # Índice de la clase predicha para el número
    suit_label = np.argmax(prediction[0][13:])   # Índice de la clase predicha para el palo
    
    # Mapear los índices de clase a los valores reales
    number = str(number_label + 1) if number_label < 9 else {9: '10', 10: 'J', 11: 'Q', 12: 'K', 13: 'A'}[number_label + 1]
    suit = {0: 'Hearts', 1: 'Diamonds', 2: 'Clubs', 3: 'Spades'}[suit_label]
    
    return number, suit

# Cargar el modelo de clasificación preentrenado
model = load_model('card_classifier.h5')

# Cargar la imagen
image = cv2.imread('cartas.png')

# Preprocesar la imagen
preprocessed_image = preprocess_image(image)

# Encontrar contornos de cartas
card_contours = find_card_contours(preprocessed_image)

# Filtrar contornos de cartas
filtered_card_contours = filter_card_contours(card_contours)

# Reconocer números y palos de cartas
detected_cards = []
for contour in filtered_card_contours:
    number, suit = recognize_card(preprocessed_image, contour, model)
    detected_cards.append((number, suit))

# Mostrar los resultados
for i, (number, suit) in enumerate(detected_cards):
    print(f"Carta {i+1}: Número {number}, Palo {suit}")

# Dibujar contornos filtrados en la imagen original
image_with_filtered_contours = image.copy()
cv2.drawContours(image_with_filtered_contours, filtered_card_contours, -1, (0, 255, 0), 2)

# Mostrar la imagen original con contornos filtrados
cv2.imshow('Image with Filtered Contours', image_with_filtered_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()
